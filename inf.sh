#!/bin/bash
DATA="${1-baffle}"
CONFIG=configs/${DATA}
source configs/${DATA}

run() {

    python src/inf.py \
    -batch_size $batch_size \
    -data_path $data_path${DATA}/ \
    -num_epochs $num_epochs \
    -device $device \
    -lr $lr \
    -up_conv_dims $up_conv_dims \
    -down_conv_dims $down_conv_dims \
    -num_features $num_features \
    -seed $seed \
    -n_classes $n_classes \
    -k_p $k_p \
    -batch_norm $batch_norm \
    -drop $drop \
    -up_drop $up_drop \
    -test $test

}


    run