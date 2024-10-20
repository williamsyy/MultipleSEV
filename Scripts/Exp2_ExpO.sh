#!/bin/bash

datasets=('adult' 'compas' 'fico' 'german' 'mimic' 'diabetes' 'headline_total')
# datasets=('compas')
models=('l2lr' 'mlp' 'gbdt')
Types=('1DFed' 'Fed')

for dataset in "${datasets[@]}"
do
    for model in "${models[@]}"
    do
        for Type in "${Types[@]}"
        do
            sbatch --wrap="python ../Experiments/Experiment\ 2\ ExpO.py --dataset $dataset --model $model --iterations 10 --Type $Type" --output="../Results/out/expo $dataset $model $Type.out" --error="../Results/err/expo $dataset $model $Type.err" --mem=200G -p compsci
            sleep 10
        done
    done
done