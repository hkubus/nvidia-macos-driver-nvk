#!/usr/bin/env python3
"""Drop flat-namespace weak-import binds from a Mach-O (x86_64) dylib's LC_DYLD_INFO bind stream.

Mesa's Vulkan dispatch tables name every entry point of every extension; the ones a driver does not implement are weak
undefined symbols, which a macOS link with -undefined dynamic_lookup turns into flat-namespace weak imports. dyld looks
each one up in every image loaded in the process and, finding none, stores NULL - for libvulkan_nouveau that is 2940
lookups across hundreds of images, ~80 ms at every process start that touches Metal. The slot already holds 0 in the
file, so removing the bind gives the same memory without the search.

The tool re-encodes the remaining binds in their original order, pads the stream to its original size (no offsets in
the file move), checks every dropped slot is 0 in the file, decodes its own output to prove it equals the original
records minus the dropped ones, and refuses on anything it does not understand.

usage: drop_flat_weak_binds.py <in.dylib> <out.dylib> [--keep-list <file of symbols that must NOT be dropped>]
"""
import struct, sys

LC_SEGMENT_64, LC_DYLD_INFO, LC_DYLD_INFO_ONLY, LC_CODE_SIGNATURE = 0x19, 0x22, 0x80000022, 0x1D
FLAT, WEAK_IMPORT, PTR = -2, 0x1, 8

def uleb(b, i):
    r = s = 0
    while True:
        c = b[i]; i += 1; r |= (c & 0x7F) << s; s += 7
        if not c & 0x80: return r, i

def sleb(b, i):
    r = s = 0
    while True:
        c = b[i]; i += 1; r |= (c & 0x7F) << s; s += 7
        if not c & 0x80:
            if c & 0x40: r -= 1 << s
            return r, i

def enc_uleb(v):
    out = bytearray()
    while True:
        c = v & 0x7F; v >>= 7
        if v: out.append(c | 0x80)
        else: out.append(c); return bytes(out)

def enc_sleb(v):
    out = bytearray()
    while True:
        c = v & 0x7F; v >>= 7
        done = (v == 0 and not c & 0x40) or (v == -1 and c & 0x40)
        out.append(c if done else c | 0x80)
        if done: return bytes(out)

def decode(b):
    """-> list of records (seg, off, ordinal, sym, flags, type, addend), in stream order."""
    recs, i, n = [], 0, len(b)
    seg = off = 0; ordinal = 0; sym = None; flags = 0; typ = 1; addend = 0
    def bind():
        recs.append((seg, off, ordinal, sym, flags, typ, addend))
    while i < n:
        c = b[i]; i += 1; op, imm = c & 0xF0, c & 0x0F
        if op == 0x00: break                                   # DONE
        elif op == 0x10: ordinal = imm                         # SET_DYLIB_ORDINAL_IMM
        elif op == 0x20: ordinal, i = uleb(b, i)               # SET_DYLIB_ORDINAL_ULEB
        elif op == 0x30: ordinal = 0 if imm == 0 else (imm | 0xF0) - 0x100   # SET_DYLIB_SPECIAL_IMM
        elif op == 0x40:                                       # SET_SYMBOL_TRAILING_FLAGS_IMM
            j = b.index(0, i); sym = bytes(b[i:j]); i = j + 1; flags = imm
        elif op == 0x50: typ = imm                             # SET_TYPE_IMM
        elif op == 0x60: addend, i = sleb(b, i)                # SET_ADDEND_SLEB
        elif op == 0x70: seg = imm; off, i = uleb(b, i)        # SET_SEGMENT_AND_OFFSET_ULEB
        elif op == 0x80: d, i = uleb(b, i); off = (off + d) & 0xFFFFFFFFFFFFFFFF   # ADD_ADDR_ULEB
        elif op == 0x90: bind(); off += PTR                    # DO_BIND
        elif op == 0xA0: bind(); d, i = uleb(b, i); off = (off + PTR + d) & 0xFFFFFFFFFFFFFFFF
        elif op == 0xB0: bind(); off += PTR + imm * PTR
        elif op == 0xC0:
            cnt, i = uleb(b, i); skip, i = uleb(b, i)
            for _ in range(cnt): bind(); off += PTR + skip
        else: raise SystemExit(f"unknown bind opcode 0x{c:02x} at {i-1}")
    return recs

def encode(recs):
    out = bytearray(); cur = dict(seg=None, off=None, ordinal=None, sym=None, flags=None, typ=None, addend=0)
    for seg, off, ordinal, sym, flags, typ, addend in recs:
        if ordinal != cur['ordinal']:
            if ordinal <= 0: out.append(0x30 | (ordinal & 0x0F))
            elif ordinal < 16: out.append(0x10 | ordinal)
            else: out.append(0x20); out += enc_uleb(ordinal)
            cur['ordinal'] = ordinal
        if sym != cur['sym'] or flags != cur['flags']:
            out.append(0x40 | flags); out += sym + b'\0'; cur['sym'], cur['flags'] = sym, flags
        if typ != cur['typ']: out.append(0x50 | typ); cur['typ'] = typ
        if addend != cur['addend']: out.append(0x60); out += enc_sleb(addend); cur['addend'] = addend
        if seg != cur['seg'] or off != cur['off']:
            out.append(0x70 | seg); out += enc_uleb(off); cur['seg'] = seg
        out.append(0x90); cur['off'] = off + PTR
    out.append(0x00)
    return bytes(out)

def main():
    a = sys.argv[1:]
    if len(a) < 2: raise SystemExit(__doc__)
    src, dst = a[0], a[1]
    keep = set()
    if '--keep-list' in a: keep = {l.strip().encode() for l in open(a[a.index('--keep-list') + 1]) if l.strip()}
    data = bytearray(open(src, 'rb').read())
    magic, cpu, _, ftype, ncmds, sizeofcmds, _, _ = struct.unpack_from('<IiiIIIII', data, 0)
    if magic != 0xFEEDFACF or cpu != 0x01000007: raise SystemExit("not a thin x86_64 Mach-O")
    p, segs, info = 32, [], None
    for _ in range(ncmds):
        cmd, size = struct.unpack_from('<II', data, p)
        if cmd == LC_SEGMENT_64:
            name, vmaddr, vmsize, fileoff, filesize = struct.unpack_from('<16sQQQQ', data, p + 8)
            segs.append((name.rstrip(b'\0').decode(), vmaddr, vmsize, fileoff, filesize))
        elif cmd in (LC_DYLD_INFO, LC_DYLD_INFO_ONLY):
            info = struct.unpack_from('<10I', data, p + 8)
        p += size
    if not info:   # chained fixups: binds are encoded differently and this tool does not rewrite them; copy unchanged
        open(dst, 'wb').write(data); print("no LC_DYLD_INFO (chained fixups?) - copied unchanged"); return
    bind_off, bind_size = info[2], info[3]
    stream = bytes(data[bind_off:bind_off + bind_size])
    recs = decode(stream)
    drop = [r for r in recs if r[2] == FLAT and (r[4] & WEAK_IMPORT) and r[3] not in keep]
    kept = [r for r in recs if not (r[2] == FLAT and (r[4] & WEAK_IMPORT) and r[3] not in keep)]
    # every dropped slot must already be 0 in the file (and carry no addend), or dyld's NULL is not what the file holds
    for seg, off, ordinal, sym, flags, typ, addend in drop:
        if typ != 1 or addend: raise SystemExit(f"dropped bind {sym} has type {typ} / addend {addend}")
        _, vmaddr, vmsize, fileoff, filesize = segs[seg]
        if off + PTR > filesize: raise SystemExit(f"slot of {sym} is outside the file part of {segs[seg][0]}")
        v = struct.unpack_from('<Q', data, fileoff + off)[0]
        if v: raise SystemExit(f"slot of {sym} holds 0x{v:x}, not 0")
    new = encode(kept)
    if len(new) > bind_size: raise SystemExit(f"re-encoded stream {len(new)} B exceeds the original {bind_size} B")
    if decode(new) != kept: raise SystemExit("re-encoded stream does not decode to the kept records")
    data[bind_off:bind_off + bind_size] = new + b'\0' * (bind_size - len(new))
    open(dst, 'wb').write(data)
    print(f"binds: {len(recs)} -> {len(kept)} (dropped {len(drop)} flat-namespace weak imports; kept {len(keep)} named);"
          f" bind stream {len(new)} of {bind_size} B")

main()
