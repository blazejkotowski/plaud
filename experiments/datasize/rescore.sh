#!/bin/zsh
# Re-evaluate every data-size checkpoint with whole-passage, mean-latent reconstruction (no 2 s seams)
# into results_whole/, then the perceptual scores. Shares the GPU with whatever trains.
#   nohup experiments/datasize/rescore.sh > ~/rescore.log 2>&1 &
cd /home/btadeusz/code/plaud
PY=/home/btadeusz/miniconda3/envs/ddsp/bin/python
export PYTHONPATH=/home/btadeusz/code/plaud
D=/mnt/mariadata/datasets/research/datasize
R=experiments/datasize/results_whole
typeset -A FULL
FULL=(harbour configs/factory3/04_harbour_nosilence.yaml percussion configs/factory2/02_percussion_nosilence.yaml
      drone_sampled configs/factory2/04_drone_sampled_nosilence.yaml)
mkdir -p $R
for s in harbour percussion drone_sampled; do
  [ -f $R/${s}_full_e160.json ] || $PY experiments/datasize/eval_recon.py --config $FULL[$s] --excerpts $D/$s/excerpts.json --ckpt last --out $R/${s}_full_e160.json
  for size in 16 128 960; do
    name=ds_${s}_s$size
    for c in training/research/synth/$name/e*.ckpt(N); do
      e=$(basename $c .ckpt); e=$(( ${e#e} + 1 ))
      [ -f $R/${name}_e$e.json ] || $PY experiments/datasize/eval_recon.py --config configs/research_datasize/${s}_s$size.yaml \
          --excerpts $D/$s/excerpts.json --ckpt $c --out $R/${name}_e$e.json
    done
  done
done
/home/btadeusz/miniconda3/envs/percept/bin/python experiments/datasize/eval_percept.py --results $R --out experiments/datasize/percept_whole.json
echo "$(date) rescore finished"
