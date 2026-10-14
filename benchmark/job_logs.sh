#!/usr/bin/env bash
# Usage: ./job_logs.sh <job-id>...  → results/<job-id>/log.txt, then analyze.py.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
for id in "$@"; do
  mkdir -p "$here/results/$id"
  hf jobs logs "$id" > "$here/results/$id/log.txt"
  echo "$id: $(wc -l < "$here/results/$id/log.txt") lines"
done
python3 "$here/analyze.py" "$here/results"
