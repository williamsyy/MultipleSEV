#!/bin/bash

datasets=('fico' 'mimic' 'diabetes' 'headline_total')
models=('l1lr' 'l2lr' 'mlp' 'gbdt')
tolerances=('0.05' '0.1' '0.2')

for dataset in "${datasets[@]}"
do
    for model in "${models[@]}"
    do
        for tolerance in "${tolerances[@]}"
        do
            sbatch --wrap="python ../Experiments/Experiment\ 1\ Flexible.py --dataset $dataset --model $model --tolerance $tolerance --iterations 10" --output="../Results/out/flexible $dataset $model $tolerance.out" --error="../Results/err/flexible $dataset $model $tolerance.err" -p compsci
        done
    done
done