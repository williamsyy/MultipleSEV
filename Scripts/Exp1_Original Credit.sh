#!/bin/bash

datasets=('adult' 'german' 'compas' 'fico' 'mimic' 'diabetes' 'headline_total')
models=('l2lr' 'l1lr' 'mlp' 'gbdt')

for dataset in "${datasets[@]}"
do
    for model in "${models[@]}"
    do
        sbatch --wrap="python ../Experiments/Experiment\ 1\ Original\ Credible.py --dataset $dataset --model $model --iterations 10" --output="../Results/out/original $dataset $model _credit.out" --error="../Results/err/original $dataset $model _credit.err" -p compsci
    done
done