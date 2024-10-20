#!/bin/bash

datasets=('adult' 'compas' 'fico' 'german' 'mimic' 'diabetes' 'headline_total')

for dataset in "${datasets[@]}"
do
    sbatch --wrap="python ../Experiments/Experiment_3_CF.py --dataset $dataset" --output="../Results/out/bcarla $dataset.out" --error="../Results/err/carla $dataset.err" --mem=200G -p compsci
done