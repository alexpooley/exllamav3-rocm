#!/bin/bash
# device assembly of the prototype, under the build lock
cd /home/alex/code/exl3/wt-q27persist/persist
F=$(grep "^cuda_cflags" build/build.ninja | sed 's/^cuda_cflags = //')
exec 8>/tmp/xtx-build.lock; flock 8
/opt/rocm/bin/hipcc $F --cuda-device-only -S persist_mlp.hip -o /tmp/pm_dev.s 2>/dev/null
python3 - <<'PY'
import re
s=open('/tmp/pm_dev.s').read()
for m in re.finditer(r'^(_ZN2pm\w+):\n(.*?)\n\s*s_endpgm', s, re.S|re.M):
    name, body = m.group(1), m.group(2)
    if 'ILi8E' not in name: continue
    lines=[l for l in body.split('\n') if l.strip() and not l.strip().startswith(('.', ';'))]
    c=lambda p: sum(1 for l in lines if re.match(r'\s*'+p, l))
    print(name[:60], 'instr', len(lines), 'buffer_load', c('buffer_load'), 'v_dot2', c('v_dot2'), 'waitcnt', c('s_waitcnt'), 'scratch', c('scratch_'), 'ds_bperm', c('ds_bpermute'), 'v_sad', c('v_sad'))
PY
