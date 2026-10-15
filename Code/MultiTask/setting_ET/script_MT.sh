#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=20:00:00
#SBATCH --output=finetune.out
 
module load 2023
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
 
# Activate your venv (built with --system-site-packages so it can see the
# module-provided torch). Located on work3 rather than $HOME due to home
# directory quota limits.
source /gpfs/work3/0/khse0533/hannah/envs/srl-env/bin/activate

learningrates=(5e-5)
epochs=(30)
batch_sizes=(16)
modelnames=('google-bert/bert-base-multilingual-cased') #'globalise/GloBERTise' #'google-bert/bert-base-multilingual-cased' #'FacebookAI/xlm-roberta-base'
model_types=(BERT) #RoBERTa #BERT #XLM
seeds=(8 51 222 246) 
 
DEST_ROOT="/gpfs/work3/0/khse0533/hannah/Documents/MD/ET/Multi/BERT/"
mkdir -p "$DEST_ROOT"
 
for learningrate in "${learningrates[@]}"
do
    for epoch in "${epochs[@]}"
    do
    for batch_size in "${batch_sizes[@]}"
        do
            for modelname in "${modelnames[@]}"
            do
                for model_type in "${model_types[@]}"
                do
                    for seed in "${seeds[@]}"
                    do
                    DEST="${DEST_ROOT}/seed${seed}/"
                    mkdir -p "$DEST"
 
                    python new_MT.py --learning_rate=$learningrate \
                                            --epoch=$epoch \
                                            --batch_size=$batch_size \
                                            --model_checkpoint=$modelname \
                                            --model_type=$model_type \
                                            --seed=$seed
 
                    cp "$TMPDIR/outputs/seed${seed}"/*.json "$DEST" 2>/dev/null
                    cp "$TMPDIR/outputs/seed${seed}"/*.csv "$DEST" 2>/dev/null
                    cp "$TMPDIR/outputs/seed${seed}"/*.txt "$DEST" 2>/dev/null
                    cp "$TMPDIR/outputs/seed${seed}"/*.png "$DEST" 2>/dev/null
 
                    # Checkpoint (already limited to one)
                    cp -r "$TMPDIR/outputs/seed${seed}"/checkpoint-* "$DEST" 2>/dev/null
 
                    done
                done
            done
        done
    done
done
 