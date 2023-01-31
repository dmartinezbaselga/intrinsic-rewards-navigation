#!/bin/bash
day=`date +%m%d`
echo "The Script begin at $day"
a=0.2
b=-0.25
c=0.25
d=1.0
# Script to reproduce results
for ((i=0;i<1;i+=1))
do
	python3.6 train.py \
	--policy tree-search-rl \
	--output_dir data/$day/tsrl_curiosity/$i \
	--randomseed $i  \
	--config configs/icra_benchmark/ts_separate_curiosity.py \
	--goal_weight $a \
	--re_collision $b \
	--re_arrival $c \
	--safe_weight $d \
	--human_num 10

done