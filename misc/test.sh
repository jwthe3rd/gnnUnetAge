#!/bin/bash


DATA="${1-DD}"

CONFIG=${DATA}

source ${DATA}

echo $num_iterations
echo $x_value
echo $start

run(){

	python input_params.py \
		-num_iterations $num_iterations \
		-x_value $x_value \
		-start $start

	}


	run
