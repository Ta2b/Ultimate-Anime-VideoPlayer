#!/usr/bin/env python3
"""Résout les conflits « additifs » : garde les deux côtés.
- listes SOURCES CMake : union triée, re-enveloppée ;
- autres blocs : côté HEAD puis lignes du côté entrant absentes de HEAD."""
import re, sys

def resolve(text, path):
    out, i = [], 0
    lines = text.split('\n')
    while i < len(lines):
        if lines[i].startswith('<<<<<<< '):
            j = i + 1; ours = []
            while not lines[j].startswith('======='): ours.append(lines[j]); j += 1
            k = j + 1; theirs = []
            while not lines[k].startswith('>>>>>>> '): theirs.append(lines[k]); k += 1
            joined = '\n'.join(ours + theirs)
            if path.endswith('CMakeLists.txt') and 'SOURCES' in joined:
                indent = re.match(r'\s*', ours[0]).group(0)
                files = sorted(set(re.findall(r'[\w./-]+\.(?:cpp|cu|c)\b', joined)))
                cur, rows = indent + 'SOURCES', []
                for f in files:
                    if len(cur) + 1 + len(f) > 100:
                        rows.append(cur); cur = indent + '        ' + f
                    else:
                        cur += ' ' + f
                tail = ')' if any(l.rstrip().endswith(')') for l in ours + theirs) else ''
                rows.append(cur + tail)
                out.extend(rows)
            else:
                key = lambda l: l.split('//')[0].strip() or l.strip()
                seen = set(key(l) for l in ours)
                out.extend(ours + [l for l in theirs if key(l) not in seen])
            i = k + 1
        else:
            out.append(lines[i]); i += 1
    return '\n'.join(out)

for p in sys.argv[1:]:
    s = open(p).read()
    open(p, 'w').write(resolve(s, p))
