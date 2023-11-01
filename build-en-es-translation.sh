#!/bin/sh

set -vex

original_dir=$(pwd)

pip install fastBPE sacremoses subword_nmt
pip install -e .

rm -rf ../apex
git clone https://github.com/NVIDIA/apex ../apex
cd ../apex
git checkout 22.04-dev

pip wheel -v -w /tmp \
  --disable-pip-version-check \
  --no-cache-dir \
  --global-option="--cpp_ext" \
  --global-option="--cuda_ext" \
  .

pip install /tmp/apex*.whl

cd $original_dir
cd examples/translation/
bash prepare-wmt14en2es.sh
cd ../..

TEXT=examples/translation/wmt17_en_es

fairseq-preprocess --source-lang en --target-lang es \
  --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
  --destdir data-bin/wmt17.en-es \
  --workers 20

CUDA_VISIBLE_DEVICES=1

fairseq-train \
    data-bin/wmt17.en-es \
    --arch transformer --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --save-dir checkpoints-en-es \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric

fairseq-generate data-bin/wmt14.en-es \
    --path checkpoints-en-es/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe

