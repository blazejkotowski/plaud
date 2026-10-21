#!/bin/zsh
# Data-size experiment runner (maria). Trains the research synths one at a time with the GPU to itself:
# while a run trains or is evaluated, every PLANC process (cwd ~/code/planc) is SIGSTOPped, and resumed after.
#   nohup experiments/datasize/run.sh > ~/datasize.log 2>&1 &
cd /home/btadeusz/code/plaud
PY=/home/btadeusz/miniconda3/envs/ddsp/bin/python
export PYTHONPATH=/home/btadeusz/code/plaud
D=/mnt/mariadata/datasets/research/datasize
R=experiments/datasize/results
SETS=(harbour percussion drone_sampled)
SIZES=(16 128 960)
typeset -A FULL
FULL=(harbour configs/factory3/04_harbour_nosilence.yaml percussion configs/factory2/02_percussion_nosilence.yaml
      drone_sampled configs/factory2/04_drone_sampled_nosilence.yaml)

planc_pids() { for p in $(pgrep python); do [ "$(readlink /proc/$p/cwd)" = /home/btadeusz/code/planc ] && echo $p; done }
pause_planc() { while true; do for p in $(planc_pids); do kill -STOP $p 2>/dev/null; done; sleep 15; done }
hold() { pause_planc & HOLD=$!; sleep 1; echo "$(date) PLANC paused: $(planc_pids | tr '\n' ' ')"; }
release() { kill $HOLD 2>/dev/null; wait $HOLD 2>/dev/null; for p in $(planc_pids); do kill -CONT $p 2>/dev/null; done; echo "$(date) PLANC resumed"; }
trap 'release; exit 1' INT TERM

for s in $SETS; do
  [ -f $D/$s/excerpts.json ] || $PY experiments/datasize/make_subsets.py --set $s --sizes ${(j:,:)SIZES} --out $D
done
$PY experiments/datasize/make_configs.py --sizes ${(j:,:)SIZES}

while pgrep -f "/factory[_]queue[.]sh" > /dev/null; do sleep 60; done
echo "$(date) GPU free of the mono PLAUD queue"
mkdir -p $R

hold
for s in $SETS; do          # the full-size point: the trained factory model
  [ -f $R/${s}_full_e160.json ] || $PY experiments/datasize/eval_recon.py --config $FULL[$s] --excerpts $D/$s/excerpts.json \
      --ckpt last --out $R/${s}_full_e160.json
done
release

for size in $SIZES; do
  for s in $SETS; do
    name=ds_${s}_s$size
    hold
    if [ ! -f training/research/synth/$name/e159.ckpt ]; then
      echo "$(date) train $name"
      SYNTH_NO_PBAR=1 $PY -m cli.train_multiple configs/research_datasize --configs ${s}_s$size --stages synth \
          --report-name train_report_${s}_s$size.txt
    fi
    for c in training/research/synth/$name/e*.ckpt; do
      e=$(basename $c .ckpt); e=$(( ${e#e} + 1 ))
      [ -f $R/${name}_e$e.json ] || $PY experiments/datasize/eval_recon.py --config configs/research_datasize/${s}_s$size.yaml \
          --excerpts $D/$s/excerpts.json --ckpt $c --out $R/${name}_e$e.json
    done
    release
  done
done
echo "$(date) data-size experiment finished"
