#!/bin/bash
source inf_config

run() {

    python inference.py \
    -batch_size $batch_size \
    -data_path $data_path \
    -num_epochs $num_epochs \
    -device $device \
    -lr $lr \
    -max_v $max_v \
    -max_L $max_L \
    -up_conv_dims $up_conv_dims \
    -down_conv_dims $down_conv_dims \
    -lat_dim $lat_dim \
    -num_features $num_features \
    -seed $seed \
    -n_classes $n_classes \
    -k_p $k_p \
    -Re_size $Re_size \
    -baffle_size $baffle_size \
    -test $test
}


    run