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

	echo "RE3"

	python3.6 test.py \
	--model_dir data/0726/tsrl_random_encoder_v3/0 \
	--config data/0726/tsrl_random_encoder_v3/0/config.py \
	--policy tree-search-rl \
	--human_num 10 \
	--circle \
	--planning_depth 1 \
	--visualize \
	--planning_width 10

	python ros_node.py \
	--model_dir data/0726/tsrl_random_encoder_v3/0 \
	--config data/0726/tsrl_random_encoder_v3/0/config.py \
	--policy tree_search_rl \
	--planning_depth 1 \
	--planning_width 10


done