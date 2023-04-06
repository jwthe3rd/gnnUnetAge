#!/bin/bash
#
#

k_p=(0.7 0.5 0.3)
depth=("64 128" "64 128 256" "64 128 256 512")
lr=(0.01 0.005 0.001)

for i in "${!k_p[@]}"; do
	for j in "${!depth[@]}"; do
		for k in "${!lr[@]}"; do

			dep=$(($j + 2))

			if [ -f "./figs/model_lr_${lr[$k]}_depth_${dep}_k_${k_p[$i]}_accuracy.png" ]; then
				echo ${k_p[$i]} ${depth[$j]} ${lr[$k]}
		
			else
				sed "15c\
				k_p=${k_p[$i]}" configs/baffle8 >configs/baffle10
				sed '7c\
down_conv_dims="'"${depth[$j]}"'"' configs/baffle10 >configs/baffle11
				sed "5c\
				lr=${lr[$k]}" configs/baffle11 >configs/baffle12

				mv configs/baffle12 configs/baffle8

				./train.sh baffle8

			fi

		done
	done
done
