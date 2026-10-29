#!/bin/bash
# 서버 부하가 낮아질 때까지 기다린다.
#
# 러너는 여러 사람이 함께 쓰는 서버에서 돈다. 부하가 높을 때 빌드를 시작하면
# 메모리 압박으로 systemd-oomd 가 러너를 죽이고 다른 사람 작업도 느려진다.
# 그래서 1분 평균 부하와 가용 메모리가 기준 안에 연속으로 들어올 때만 진행한다.
#
# 컨테이너 안의 /proc/loadavg, /proc/meminfo 는 호스트 값을 그대로 보여준다.
#
# 환경 변수(저장소 Variables 로 덮어쓸 수 있다):
#   MAX_LOAD      1분 평균 부하 상한. 기본값: 코어 수의 절반
#   MIN_MEM_GB    가용 메모리 하한(GB). 기본값: 64
#   STABLE_CHECKS 기준을 연속으로 만족해야 하는 횟수(1분 간격). 기본값: 3
#   MAX_WAIT_MIN  최대 대기 시간(분). 넘으면 실패한다. 기본값: 180
set -euo pipefail

cores=$(nproc)
max_load=${MAX_LOAD:-$((cores / 2))}
min_mem_gb=${MIN_MEM_GB:-64}
stable_checks=${STABLE_CHECKS:-3}
max_wait_min=${MAX_WAIT_MIN:-180}

echo "기준: 1분 부하 < ${max_load} (코어 ${cores}), 가용 메모리 >= ${min_mem_gb}GB, ${stable_checks}분 연속"

ok=0
for ((min = 0; min <= max_wait_min; min++)); do
  load=$(cut -d' ' -f1 /proc/loadavg)
  mem_gb=$(awk '/^MemAvailable:/ {printf "%d", $2 / 1048576}' /proc/meminfo)

  if awk -v l="$load" -v m="$max_load" 'BEGIN { exit !(l < m) }' && ((mem_gb >= min_mem_gb)); then
    ok=$((ok + 1))
    status="ok ${ok}/${stable_checks}"
  else
    ok=0
    status="대기"
  fi
  echo "[$(date -u +%H:%M:%S)] 부하 ${load}, 가용 메모리 ${mem_gb}GB → ${status}"

  if ((ok >= stable_checks)); then
    echo "부하가 낮다. 빌드를 시작한다."
    exit 0
  fi
  sleep 60
done

echo "::error::서버 부하가 ${max_wait_min}분 동안 기준 아래로 내려오지 않아 빌드를 시작하지 않았다. 나중에 Re-run 하면 된다."
exit 1
