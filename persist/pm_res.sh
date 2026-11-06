#!/bin/bash
# kernel resource usage of the prototype (device-only compile), under the build lock
cd /home/alex/code/exl3/wt-q27persist/persist
F=$(grep "^cuda_cflags" build/build.ninja | sed 's/^cuda_cflags = //')
exec 8>/tmp/xtx-build.lock; flock 8
/opt/rocm/bin/hipcc $F --cuda-device-only -c persist_mlp.hip -o /tmp/pm_dev.o -Rpass-analysis=kernel-resource-usage 2>&1 | grep -E "Function Name|VGPRs:|SGPRs:|LDS Size|Occupancy|ScratchSize" | sed 's/.*remark: //'
