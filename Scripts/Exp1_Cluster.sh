#!/bin/bash

# datasets=('adult' 'compas' 'fico' 'german' 'mimic' 'diabetes' 'headline_total')
datasets=('diabetes' 'headline_total' 'fico' 'mimic')
models=('l2lr' 'l1lr' 'mlp' 'gbdt')

for dataset in "${datasets[@]}"
do
    for model in "${models[@]}"
    do
        sbatch --wrap="python ../Experiments/Experiment\ 1\ Cluster.py --dataset $dataset --model $model --iterations 10" --output="../Results/out/cluster $dataset $model.out" --error="../Results/err/cluster $dataset $model.err" -p compsci-gpu
    done
done