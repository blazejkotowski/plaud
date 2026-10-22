#!/bin/zsh
# Style experiments runner (maria): E1 oracle, E3 lat4, E2 k12, E4 rangeadv on one set, one at a time with
# the GPU to itself. The PLANC queue is stopped for the whole series (a SIGSTOPped PLANC codec still holds
# ~15 GB of GPU memory) and restarted at the end; it resumes from its last.ckpt.
# After each run: passage reconstructions of every checkpoint, then the scores (percept env).
#   nohup experiments/styles/run.sh percussion [exp ...] > ~/styles.log 2>&1 &
# KEEP_HOLD=1 leaves PLANC stopped at the end (for a queue of several runner calls).
set=${1:-percussion}
exps=(${@[2,-1]}); (( $#exps )) || exps=(oracle lat4 k12 rangeadv)
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
hold() {
  for p in $(pgrep -f "/home/btadeusz/planc_[q]ueue.sh"); do kill $p; done
  for p in $(planc_pids); do kill $p; done; sleep 10
  for p in $(planc_pids); do kill -9 $p; done
  echo "$(date) PLANC stopped"
}
release() {
  pgrep -f "/home/btadeusz/planc_[q]ueue.sh" > /dev/null || { nohup zsh /home/btadeusz/planc_queue.sh >> /home/btadeusz/planc_queue.log 2>&1 &; }
  echo "$(date) PLANC restarted"
}
trap 'release; exit 1' INT TERM
hold

for mode in oracle k12; do
  [ -f experiments/styles/labels/${set}_$mode.npy ] || $PY experiments/styles/make_styles.py --dataset $DATA \
      --mode ${mode/k12/cluster} --k 12 --out experiments/styles/labels/${set}_$mode
done
$PY experiments/styles/make_configs.py --set $set

[ -d $R/base/e160 ] || $PY experiments/styles/eval_passages.py --config $FULL[$set] --excerpts $EX \
    --ckpt $(ls training/synth/$(grep -m1 "^  name:" $FULL[$set] | awk '{print $2}')/last.ckpt) --out $R/base/e160

for exp in $exps; do
  name=st_${set}_$exp
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
  $PERCEPT experiments/styles/score_passages.py --results $R 2>/dev/null | tail -n +1 > $R/summary_after_$exp.txt
  echo "$(date) scored after $exp"; cat $R/summary_after_$exp.txt
done
[ -n "$KEEP_HOLD" ] || release
echo "$(date) style experiments finished ($set)"
