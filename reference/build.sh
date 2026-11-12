#!/usr/bin/env bash
# Build the reference train-loading model from the raw files.
#
#     reference/build.sh DATA OUT          # every step, in order
#     reference/build.sh DATA OUT assign   # one step: timetables, loading, means or assign
#
# DATA is the folder holding the inputs listed in reference/README.md; OUT receives the
# rendered configs, logs, caches, period matrices and the assignment output.
set -euo pipefail
if [[ $# -lt 2 ]]; then
  echo "usage: $0 DATA OUT [timetables|loading|means|assign]" >&2
  exit 2
fi
HERE=$(cd "$(dirname "$0")" && pwd)
DATA=$(cd "$1" && pwd)
mkdir -p "$2"
OUT=$(cd "$2" && pwd)
STEP=${3:-all}
OPS=(northern awc emr greater_anglia gtr gwr southeastern swr tpe)
export PYDANTIC_DISABLE_PLUGINS=__all__
mkdir -p "$OUT/cfg" "$OUT/logs"

render() {  # render configs/NAME.yml into OUT/cfg with DATA and OUT filled in
  sed -e "s#{DATA}#$DATA#g" -e "s#{OUT}#$OUT#g" "$HERE/configs/$1.yml" > "$OUT/cfg/$1.yml"
}

if [[ $STEP == all || $STEP == timetables ]]; then
  render timetable_sep
  render timetable_may26
  ortla timetable "$OUT/cfg/timetable_sep.yml" > "$OUT/logs/timetable_sep.log" 2>&1 &
  ortla timetable "$OUT/cfg/timetable_may26.yml" > "$OUT/logs/timetable_may26.log" 2>&1
  wait
fi

if [[ $STEP == all || $STEP == loading ]]; then
  for op in "${OPS[@]}"; do render "loading_$op"; done
  # The first run writes the ODM cache; the others reuse it, three at a time.
  ortla loading "$OUT/cfg/loading_northern.yml" > "$OUT/logs/loading_northern.log" 2>&1
  printf '%s\n' "${OPS[@]:1}" | xargs -P 3 -I{} \
    sh -c "ortla loading '$OUT/cfg/loading_{}.yml' > '$OUT/logs/loading_{}.log' 2>&1"
fi

if [[ $STEP == all || $STEP == means ]]; then
  python3 "$HERE/means.py" "$DATA" "$OUT" > "$OUT/logs/means.log" 2>&1
fi

if [[ $STEP == all || $STEP == assign ]]; then
  render reference
  ortla assign "$OUT/cfg/reference.yml" > "$OUT/logs/assign.log" 2>&1
fi
