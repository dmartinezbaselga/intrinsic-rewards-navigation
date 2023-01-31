#!/bin/bash
day=`date +%m%d`
echo "The Script begin at $day"
# Script to reproduce results
for ((i=0;i<3;i+=1))
do 
	python train.py \
	--policy model-predictive-rl \
	--output_dir data/$day/mprl/$i \
	--randomseed $i  \
	--config configs/icra_benchmark/mp_separate.py
done
