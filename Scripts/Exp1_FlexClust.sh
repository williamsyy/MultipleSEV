#!/bin/bash

datasets=('adult' 'compas' 'fico' 'german' 'mimic' 'diabetes' 'headline_total')
models=('l1lr' 'l2lr' 'mlp' 'gbdt')
tolerances=('0.05' '0.1' '0.2')
# dataset=('compas')


for dataset in "${datasets[@]}"
do
    for model in "${models[@]}"
    do
        for tolerance in "${tolerances[@]}"
        do
            sbatch --wrap="python ../Experiments/Experiment\ 1\ FlexClust.py --dataset $dataset --model $model --tolerance $tolerance --iterations 10" --output="../Results/out/flexclust $dataset $model $tolerance.out" --error="../Results/err/flexclust $dataset $model $tolerance.err" -p compsci-gpu
        done
    done
done