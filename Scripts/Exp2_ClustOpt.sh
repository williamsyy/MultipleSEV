#!/bin/bash

datasets=('adult' 'compas' 'fico' 'german' 'mimic' 'diabetes' 'headline_total')
# datasets=('compas')
models=('l2lr' 'mlp' 'gbdt')

for dataset in "${datasets[@]}"
do
    for model in "${models[@]}"
    do
        sbatch --wrap="python ../Experiments/Experiment\ 2\ AllOptC.py --dataset $dataset --model $model --iterations 10" --output="../Results/out/alloptc $dataset $model.out" --error="../Results/err/alloptc $dataset $model.err" --mem=100G -p compsci
    done
done