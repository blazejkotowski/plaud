#!/bin/zsh
# Style experiments runner (maria): E1 oracle, E3 lat4, E2 k12, E4 rangeadv on one set, one at a time with
# the GPU to itself (PLANC processes, cwd ~/code/planc, are SIGSTOPped while a run trains or is evaluated).
# After each run: passage reconstructions of every checkpoint, then the scores (percept env).
#   nohup experiments/styles/run.sh percussion > ~/styles.log 2>&1 &
set=${1:-percussion}
cd /home/btadeusz/code/plaud
PY=/home/btadeusz/miniconda3/envs/ddsp/bin/python
PERCEPT=/home/btadeusz/miniconda3/envs/percept/bin/python
export PYTHONPATH=/home/btadeusz/code/plaud
EX=/mnt/mariadata/datasets/research/datasize/$set/excerpts.json
R=experiments/styles/results/$set
typeset -A FULL
FULL=(harbour configs/factory3/04_harbour_nosilence.yaml percussion configs/factory2/02_percussion_nosilence.yaml
      drone_sampled configs/factory2/04_drone_sampled_nosilence.yaml)
DATA=$(grep -m1 "^  dataset_path:" $FULL[$set] | awk '{print $2}')

planc_pids() { for p in $(pgrep python); do [ "$(readlink /proc/$p/cwd)" = /home/btadeusz/code/planc ] && echo $p; done }
pause_planc() { while true; do for p in $(planc_pids); do kill -STOP $p 2>/dev/null; done; sleep 15; done }
hold() { pause_planc & HOLD=$!; sleep 1; echo "$(date) PLANC paused"; }
release() { kill $HOLD 2>/dev/null; wait $HOLD 2>/dev/null; for p in $(planc_pids); do kill -CONT $p 2>/dev/null; done; echo "$(date) PLANC resumed"; }
trap 'release; exit 1' INT TERM

for mode in oracle k12; do
  [ -f experiments/styles/labels/${set}_$mode.npy ] || $PY experiments/styles/make_styles.py --dataset $DATA \
      --mode ${mode/k12/cluster} --k 12 --out experiments/styles/labels/${set}_$mode
done
$PY experiments/styles/make_configs.py --set $set

hold
[ -d $R/base/e160 ] || $PY experiments/styles/eval_passages.py --config $FULL[$set] --excerpts $EX \
    --ckpt $(ls training/synth/$(grep -m1 "^  name:" $FULL[$set] | awk '{print $2}')/last.ckpt) --out $R/base/e160
release

for exp in oracle lat4 k12 rangeadv; do
  name=st_${set}_$exp
  hold
  if [ ! -f training/research/synth/$name/e159.ckpt ]; then
    echo "$(date) train $name"
    SYNTH_NO_PBAR=1 $PY -m cli.train_multiple configs/research_styles --configs ${set}_$exp --stages synth \
        --report-name train_report_${set}_$exp.txt
  fi
  for c in training/research/synth/$name/e*.ckpt(N); do
    e=$(basename $c .ckpt); e=$(( ${e#e} + 1 ))
    [ -d $R/$exp/e$e ] || $PY experiments/styles/eval_passages.py --config configs/research_styles/${set}_$exp.yaml \
        --excerpts $EX --ckpt $c --out $R/$exp/e$e
  done
  release
  $PERCEPT experiments/styles/score_passages.py --results $R 2>/dev/null | tail -n +1 > $R/summary_after_$exp.txt
  echo "$(date) scored after $exp"; cat $R/summary_after_$exp.txt
done
echo "$(date) style experiments finished ($set)"
