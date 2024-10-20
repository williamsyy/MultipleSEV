#!/bin/bash

datasets=('adult' 'compas' 'fico' 'german' 'mimic' 'diabetes' 'headline_total')
models=('l1lr' 'l2lr' 'mlp' 'gbdt')
tolerances=('0.05' '0.1' '0.2')

for dataset in "${datasets[@]}"
do
    for model in "${models[@]}"
    do
        for tolerance in "${tolerances[@]}"
        do
            sbatch --wrap="python ../Experiments/Experiment\ 2\ AllOptF.py --dataset $dataset --model $model --tolerance $tolerance --iterations 10" --output="../Results/out/flexopt $dataset $model $tolerance.out" --error="../Results/err/flexopt $dataset $model $tolerance.err" --mem=100G -n16 -p compsci
        done
    done
done