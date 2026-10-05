"""Minimal SPSS .sav reader (uncompressed / bytecode-compressed), numpy only.
Returns (df-like dict of numpy arrays, meta)."""
import struct
import numpy as np


def read_sav(path):
    b = open(path, 'rb').read()
    assert b[:4] in (b'$FL2',), b[:4]
    layout, nominal, compression, weight, ncases = struct.unpack('<5i', b[64:84])
    bias = struct.unpack('<d', b[84:92])[0]
    meta = {'product': b[4:64].decode('latin1').strip(), 'compression': compression, 'ncases': ncases,
            'created': b[92:101].decode('latin1') + ' ' + b[101:109].decode('latin1'),
            'file_label': b[109:173]}
    pos = 176
    slots = []          # each 8-byte slot: (var_index or None for continuation)
    variables = []      # dicts: name, type(width), label(bytes), missing
    value_labels = []   # (list of (rawvalue, labelbytes), [var slot indices 1-based])
    ext = {}

    def i32():
        nonlocal pos
        v = struct.unpack('<i', b[pos:pos + 4])[0]
        pos += 4
        return v

    while True:
        rt = i32()
        if rt == 2:
            typ, has_label, n_miss, pfmt, wfmt = struct.unpack('<5i', b[pos:pos + 20]); pos += 20
            name = b[pos:pos + 8]; pos += 8
            label = b''
            if has_label:
                ln = i32()
                label = b[pos:pos + ln]
                pos += (ln + 3) // 4 * 4
            miss = []
            for _ in range(abs(n_miss)):
                miss.append(struct.unpack('<d', b[pos:pos + 8])[0]); pos += 8
            if typ == -1:
                slots.append(None)
            else:
                variables.append({'short': name, 'type': typ, 'label': label, 'missing': miss, 'n_miss': n_miss,
                                  'slot': len(slots), 'print': pfmt})
                slots.append(len(variables) - 1)
        elif rt == 3:
            cnt = i32()
            labs = []
            for _ in range(cnt):
                raw = b[pos:pos + 8]; pos += 8
                ln = b[pos]; pos += 1
                lab = b[pos:pos + ln]
                pos += ((ln + 1 + 7) // 8 * 8) - 1
                labs.append((raw, lab))
            rt4 = i32(); assert rt4 == 4
            nv = i32()
            idx = list(struct.unpack('<%di' % nv, b[pos:pos + 4 * nv])); pos += 4 * nv
            value_labels.append((labs, idx))
        elif rt == 6:
            n = i32(); pos += 80 * n
        elif rt == 7:
            sub, size, count = struct.unpack('<3i', b[pos:pos + 12]); pos += 12
            ext.setdefault(sub, []).append(b[pos:pos + size * count]); pos += size * count
        elif rt == 999:
            i32()
            break
        else:
            raise ValueError('unknown record %d at %d' % (rt, pos))

    enc = 'utf-8'
    if 20 in ext:
        enc = ext[20][0].decode('ascii').strip() or 'utf-8'
    meta['encoding'] = enc

    def dec(x):
        try:
            return x.decode(enc).rstrip()
        except Exception:
            return x.decode('gbk', 'replace').rstrip()

    longnames = {}
    if 13 in ext:
        for pair in dec(ext[13][0]).split('\t'):
            if '=' in pair:
                s, l = pair.split('=', 1)
                longnames[s.strip()] = l.strip()
    for v in variables:
        s = dec(v['short']).strip()
        v['name'] = longnames.get(s, s)
        v['label'] = dec(v['label'])

    nslots = len(slots)
    # ---- data
    data = np.full((ncases, nslots), np.nan)
    strs = np.empty((ncases, nslots), dtype=object)
    sysmis = -struct.unpack('<d', struct.pack('<Q', 0x7FEFFFFFFFFFFFFF))[0]
    if compression == 0:
        for c in range(ncases):
            for s in range(nslots):
                raw = b[pos:pos + 8]; pos += 8
                data[c, s] = struct.unpack('<d', raw)[0]; strs[c, s] = raw
    elif compression == 1:
        c = s = 0
        done = False
        while not done and c < ncases:
            cmds = b[pos:pos + 8]; pos += 8
            for code in cmds:
                if c >= ncases:
                    break
                if code == 0:
                    continue
                if code == 252:
                    done = True; break
                if 1 <= code <= 251:
                    data[c, s] = code - bias; strs[c, s] = None
                elif code == 253:
                    raw = b[pos:pos + 8]; pos += 8
                    data[c, s] = struct.unpack('<d', raw)[0]; strs[c, s] = raw
                elif code == 254:
                    strs[c, s] = b' ' * 8
                elif code == 255:
                    data[c, s] = np.nan; strs[c, s] = None
                s += 1
                if s == nslots:
                    s = 0; c += 1
    else:
        raise NotImplementedError('zlib-compressed sav')
    data[data == sysmis] = np.nan
    data[np.abs(data) > 1e300] = np.nan

    out = {}
    for v in variables:
        sl = v['slot']
        if v['type'] == 0:
            col = data[:, sl].copy()
            if v['n_miss'] > 0:
                for mv in v['missing']:
                    col[col == mv] = np.nan
            out[v['name']] = col
        else:
            nsl = (v['type'] + 7) // 8
            vals = []
            for c in range(ncases):
                raw = b''.join((strs[c, sl + k] or b'        ') if isinstance(strs[c, sl + k], bytes) else b'        '
                               for k in range(nsl))
                vals.append(dec(raw[:v['type']]))
            out[v['name']] = np.array(vals, dtype=object)
    # value labels by variable name
    vl = {}
    for labs, idx in value_labels:
        for i in idx:
            vi = slots[i - 1]
            if vi is None:
                continue
            v = variables[vi]
            d = {}
            for raw, lab in labs:
                key = struct.unpack('<d', raw)[0] if v['type'] == 0 else dec(raw)
                d[key] = dec(lab)
            vl[v['name']] = d
    meta['variables'] = [{'name': v['name'], 'type': v['type'], 'label': v['label']} for v in variables]
    meta['value_labels'] = vl
    return out, meta
