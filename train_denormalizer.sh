#!/bin/bash

usage="
USAGE:
bash train_denormalizer.sh [-h] [-l language] [-s source_suff] [-t target_suff] [-o outdir] [-a arch] [-f] <DATADIR>

where:
    -h  show this help text
    -l  set the language (default: en; alternative: ru)
    -s  set the source-side suffix (default: norm)
    -t  set the target-side suffix (default: pnct)
    -o  set the output directory (default: denorm)
    -a  set the model architecture (default: large; alternative: small)
    -f  force overwriting of existing files
"

while getopts ":hl:s:t:o:a:f" opt; do
  case ${opt} in
    h ) echo $usage
      ;;
    l ) lang=$OPTARG
      ;;
    s ) src=$OPTARG
      ;;
    t ) tgt=$OPTARG
      ;;
    o ) outdir=$OPTARG
      ;;
    a ) size=$OPTARG
      ;;
    f ) force_overwrite=true
      ;;
    \? ) echo "$usage"
      ;;
  esac
done

shift $((OPTIND - 1))

datadir=$1
lang=${lang:-en}
src=${src:-norm}
tgt=${tgt:-pnct}
outdir=${outdir:-denorm}
size=${arch:-large}
force_overwrite=${force_overwrite:-false}

if [ ${size} = large ]; then
  arch=transformer_normalizer
elif [ ${size} = small ]; then
  arch=transformer_small
else
  echo "Invalid architecture. Choose between 'large' and 'small'."
  exit 1;
fi

train=train
val=valid
test=test

if [ -e ${outdir} ] && [ ${force_overwrite} = false ]; then
  echo "The directory '${outdir}' already exists. Choose another directory name with the option -o, or use the option -f to force overwriting."
  exit 1;
fi

if [ -e ${outdir} ] && [ ${force_overwrite} = true ]; then
  rm -r ${outdir}
  rm -r data-bin/${outdir}
fi

mkdir -p ${outdir}

# Process the raw en_with_types data
python3 scripts/generate_raw.py ${datadir} > ${outdir}/${tgt}.raw.txt \
        2> ${outdir}/${src}.raw.txt
# Preprocess dataset
python3 scripts/preprocess_${lang}.py ${outdir}/${src}.raw.txt \
        ${outdir}/${tgt}.raw.txt ${outdir}/${src}.prep.txt ${outdir}/${tgt}.prep.txt
# Split the parallel corpora into train, valid and test blocks
python3 scripts/split_tr_val_te.py ${outdir}/${src}.prep.txt \
        ${outdir}/${tgt}.prep.txt ${outdir} raw.${src} raw.${tgt}

# BPE-encode the corpus
# Generate a single combined corpus for the
# purpose of computing global subword units.
cat ${outdir}/train.raw.${src} ${outdir}/train.raw.${tgt} | shuf > ${outdir}/train.raw.full

# Learn the subword units in the normal way
echo "learn bpe on ${outdir}/train.raw.full ..."

subword-nmt learn-bpe -s 10000 < ${outdir}/train.raw.full > bpe_code

rm ${outdir}/train.raw.full

# Apply the bpe result to segment the source and target texts
for l in ${src} ${tgt}; do
  for set in ${train} ${val} ${test}; do
    f=${outdir}/${set}.raw.${l}
    bpe=${outdir}/${set}.bpe.${l}
    echo "apply bpe to ${f} ..."
    subword-nmt apply-bpe -c bpe_code < ${f} > ${bpe}
  done
done


# Preprocess the inputs in the standard way
fairseq-preprocess --joined-dictionary \
             --source-lang ${src} \
             --target-lang ${tgt} \
             --trainpref ${outdir}/train.bpe \
             --validpref ${outdir}/valid.bpe \
             --testpref ${outdir}/test.bpe \
             --destdir data-bin/${outdir} \
             --thresholdtgt 0 \
             --thresholdsrc 0 \
             --workers 25

# Stick the code where it is expected
mv bpe_code data-bin/${outdir}/bpe_code;

python scripts/custom_arch.py data-bin/${outdir} \
        --source-lang ${src} \
        --target-lang ${tgt} \
        --arch ${arch} \
        --share-all-embeddings \
        --dropout 0.3 \
        --weight-decay 0.0 \
        --criterion label_smoothed_cross_entropy \
        --label-smoothing 0.1 \
        --optimizer adam \
        --adam-betas '(0.9, 0.98)' \
        --clip-norm 0.0 \
        --lr 0.001 \
        --lr-scheduler inverse_sqrt \
        --warmup-updates 4000 \
        --max-tokens 3584 \
        --update-freq 16 \
        --max-update 100000 \
        --max-epoch 10 \
        --save-dir checkpoints/${outdir} \
        --skip-invalid-size-inputs-valid-test


if [ -e fairseq/scripts/average_checkpoints.py ]; then
  python3 fairseq/scripts/average_checkpoints.py \
          --inputs checkpoints/${outdir} \
          --num-epoch-checkpoints 3 \
          --output checkpoints/${outdir}/checkpoint_avg.pt
fi

echo "DONE TRAINING!"
