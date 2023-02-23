#!/bin/bash
DATA="${1-baffle}"
CONFIG=configs/${DATA}
source winconfigs/${DATA}

run() {

    python src/main.py \
    -batch_size $batch_size \
    -data_path $data_path${DATA}/ \
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
    -Re_size $Re_size \
    -baffle_size $baffle_size \
    -k_p $k_p \
    -batch_norm $batch_norm \
    -down_drop $down_drop \
    -up_drop $up_drop

}


    run