#!/bin/bash

datasets=('adult' 'compas' 'fico' 'german' 'mimic' 'diabetes' 'headline_total')
models=('l2lr' 'mlp' 'gbdt')

for dataset in "${datasets[@]}"
do
    for model in "${models[@]}"
    do
        sbatch --wrap="python ../Experiments/Experiment\ 2\ AllOpt1.py --dataset $dataset --model $model --iterations 10" --output="../Results/out/allopt1 $dataset $model.out" --error="../Results/err/allopt1 $dataset $model.err" --mem=100G -n16 -p compsci
    done
done