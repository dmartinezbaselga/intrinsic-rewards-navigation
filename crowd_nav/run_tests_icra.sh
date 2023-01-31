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

	echo "ORCA"
	python3.6 test.py \
	--config configs/icra_benchmark/orca.py \
	--policy orca \
	--human_num 10 \
	--circle 

	echo "ORCA"
	python3.6 test.py \
	--config configs/icra_benchmark/orca.py \
	--policy orca \
	--human_num 10 \
	--square

	# echo "ORIGINAL"

	# python3.6 test.py \
	# --model_dir data/0614/tsrl/0 \
	# --config data/0614/tsrl/0/config.py \
	# --policy tree-search-rl \
	# --human_num 10 \
	# --circle \
	# --planning_depth 1 \
	# --planning_width 10

	# echo "NOISY"

	# python3.6 test.py \
	# --model_dir data/0625/tsrl_noisy/0 \
	# --config data/0625/tsrl_noisy/0/config.py \
	# --policy tree-search-rl \
	# --human_num 10 \
	# --circle \
	# --planning_depth 1 \
	# --planning_width 10

	# echo "DROPOUT"

	# python3.6 test.py \
	# --model_dir data/0625/tsrl_dropout_v2/0 \
	# --config data/0625/tsrl_dropout_v2/0/config.py \
	# --policy tree-search-rl \
	# --human_num 10 \
	# --circle \
	# --planning_depth 1 \
	# --planning_width 10

	# echo "CURIOSITY"

	# python3.6 test.py \
	# --model_dir data/0716/tsrl_curiosity_v2/0 \
	# --config data/0716/tsrl_curiosity_v2/0/config.py \
	# --policy tree-search-rl \
	# --human_num 10 \
	# --circle \
	# --planning_depth 1 \
	# --planning_width 10

	# echo "RE3"

	# python3.6 test.py \
	# --model_dir data/0726/tsrl_random_encoder_v3/0 \
	# --config data/0726/tsrl_random_encoder_v3/0/config.py \
	# --policy tree-search-rl \
	# --human_num 10 \
	# --circle \
	# --planning_depth 1 \
	# --planning_width 10

	# echo "ORIGINAL"

	# python3.6 test.py \
	# --model_dir data/0614/tsrl/0 \
	# --config data/0614/tsrl/0/config.py \
	# --policy tree-search-rl \
	# --human_num 10 \
	# --square \
	# --planning_depth 1 \
	# --planning_width 10

	# echo "NOISY"

	# python3.6 test.py \
	# --model_dir data/0625/tsrl_noisy/0 \
	# --config data/0625/tsrl_noisy/0/config.py \
	# --policy tree-search-rl \
	# --human_num 10 \
	# --square \
	# --planning_depth 1 \
	# --planning_width 10

	# echo "DROPOUT"

	# python3.6 test.py \
	# --model_dir data/0625/tsrl_dropout_v2/0 \
	# --config data/0625/tsrl_dropout_v2/0/config.py \
	# --policy tree-search-rl \
	# --human_num 10 \
	# --square \
	# --planning_depth 1 \
	# --planning_width 10

	# echo "CURIOSITY"

	# python3.6 test.py \
	# --model_dir data/0716/tsrl_curiosity_v2/0 \
	# --config data/0716/tsrl_curiosity_v2/0/config.py \
	# --policy tree-search-rl \
	# --human_num 10 \
	# --square \
	# --planning_depth 1 \
	# --planning_width 10

	# echo "RE3"

	# python3.6 test.py \
	# --model_dir data/0726/tsrl_random_encoder_v3/0 \
	# --config data/0726/tsrl_random_encoder_v3/0/config.py \
	# --policy tree-search-rl \
	# --human_num 10 \
	# --square \
	# --planning_depth 1 \
	# --planning_width 10

	# echo "LSTM-RL"

	# python3.6 test.py \
	# --model_dir data/0823/lstm_rl/0 \
	# --config data/0823/lstm_rl/0/config.py \
	# --policy lstm_rl \
	# --human_num 10 \
	# --circle 

	# echo "LSTM-RL-C"

	# python3.6 test.py \
	# --model_dir data/0823/lstm_rl_curiosity/0 \
	# --config data/0823/lstm_rl_curiosity/0/config.py \
	# --policy lstm_rl \
	# --human_num 10 \
	# --circle 
	# python3.6 test.py \
	# --model_dir data/0825/lstm_rl_curiosity_v2/0 \
	# --config data/0825/lstm_rl_curiosity_v2/0/config.py \
	# --policy lstm_rl \
	# --human_num 10 \
	# --circle 

	# echo "SARL"

	# python3.6 test.py \
	# --model_dir data/0830/sarl_random_encoder/0 \
	# --config data/0830/sarl_random_encoder/0/config.py \
	# --policy sarl \
	# --human_num 10 \
	# --circle 
	# python3.6 test.py \
	# --model_dir data/0823/sarl/0 \
	# --config data/0823/sarl/0/config.py \
	# --policy sarl \
	# --human_num 10 \
	# --circle 

	# echo "SARL-C"

	# python3.6 test.py \
	# --model_dir data/0823/sarl_curiosity/0 \
	# --config data/0823/sarl_curiosity/0/config.py \
	# --policy sarl \
	# --human_num 10 \
	# --circle 
	# python3.6 test.py \
	# --model_dir data/0825/sarl_curiosity_v2/0 \
	# --config data/0825/sarl_curiosity_v2/0/config.py \
	# --policy sarl \
	# --human_num 10 \
	# --circle 

	# echo "MPRGL"

	# python3.6 test.py \
	# --model_dir data/0823/mp_separate_dp/0 \
	# --config data/0823/mp_separate_dp/0/config.py \
	# --policy model-predictive-rl \
	# --human_num 10 \
	# --circle \
	# --planning_depth 1 \
	# --planning_width 10

	# echo "MPRGL-C"

	# python3.6 test.py \
	# --model_dir data/0825/mp_separate_dp_curiosity_v2/0 \
	# --config data/0825/mp_separate_dp_curiosity_v2/0/config.py \
	# --policy model-predictive-rl \
	# --human_num 10 \
	# --circle \
	# --planning_depth 1 \
	# --planning_width 10

done