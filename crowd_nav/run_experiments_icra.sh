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
	--output_dir data/$day/tsrl_random_encoder/$i \
	--randomseed $i  \
	--config configs/icra_benchmark/ts_separate_random_encoder.py \
	--goal_weight $a \
	--re_collision $b \
	--re_arrival $c \
	--safe_weight $d \
	--human_num 10

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

	python3.6 train.py \
	--policy lstm_rl \
	--output_dir data/$day/lstm_rl_noisy/$i \
	--randomseed $i  \
	--config configs/icra_benchmark/lstm_rl_noisy.py \
	--goal_weight $a \
	--re_collision $b \
	--re_arrival $c \
	--safe_weight $d \
	--human_num 10 

	python3.6 train.py \
	--policy lstm_rl \
	--output_dir data/$day/lstm_rl_dropout/$i \
	--randomseed $i  \
	--config configs/icra_benchmark/lstm_rl_dropout.py \
	--goal_weight $a \
	--re_collision $b \
	--re_arrival $c \
	--safe_weight $d \
	--human_num 10 

	python3.6 train.py \
	--policy sarl \
	--output_dir data/$day/sarl_noisy/$i \
	--randomseed $i  \
	--config configs/icra_benchmark/sarl_noisy.py \
	--goal_weight $a \
	--re_collision $b \
	--re_arrival $c \
	--safe_weight $d \
	--human_num 10 

	python3.6 train.py \
	--policy sarl \
	--output_dir data/$day/sarl_dropout/$i \
	--randomseed $i  \
	--config configs/icra_benchmark/sarl_dropout.py \
	--goal_weight $a \
	--re_collision $b \
	--re_arrival $c \
	--safe_weight $d \
	--human_num 10 

	python3.6 train.py \
	--policy model_predictive_rl \
	--output_dir data/$day/mp_separate_dp_noisy/$i \
	--randomseed $i  \
	--config configs/icra_benchmark/mp_separate_dp_noisy.py \
	--goal_weight $a \
	--re_collision $b \
	--re_arrival $c \
	--safe_weight $d \
	--human_num 10 

	python3.6 train.py \
	--policy model_predictive_rl \
	--output_dir data/$day/mp_separate_dp_dropout/$i \
	--randomseed $i  \
	--config configs/icra_benchmark/mp_separate_dp_dropout.py \
	--goal_weight $a \
	--re_collision $b \
	--re_arrival $c \
	--safe_weight $d \
	--human_num 10 
done