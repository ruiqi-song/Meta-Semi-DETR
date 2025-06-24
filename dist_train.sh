#!/bin/bash
###
# @brief:
# @Version: v1.0.0
# @Author: knightdby  && knightdby@163.com
# @Date: 2025-03-07 14:28:02
# @Description:
# @LastEditors: knightdby
# @LastEditTime: 2025-06-24 09:07:27
# @FilePath: /MetaSemiDetr/dist_train.sh
# Copyright 2025  by Inc, All Rights Reserved.
# 2025-03-07 14:28:02
###

rangeStart=29620
rangeEnd=29630
PORT=0
function Listening {
    TCPListeningnum=$(netstat -an | grep ":$1 " | awk '$1 == "tcp" && $NF == "LISTEN" {print $0}' | wc -l)
    UDPListeningnum=$(netstat -an | grep ":$1 " | awk '$1 == "udp" && $NF == "0.0.0.0:*" {print $0}' | wc -l)
    ((Listeningnum = TCPListeningnum + UDPListeningnum))
    if [ $Listeningnum == 0 ]; then
        echo "0"
    else
        echo "1"
    fi
}
function random_range {
    shuf -i $1-$2 -n1
}
function get_random_port {
    templ=0
    while [ $PORT == 0 ]; do
        temp1=$(random_range $1 $2)
        if [ $(Listening $temp1) == 0 ]; then
            PORT=$temp1
        fi
    done
    echo "Using Port=$PORT"
}
get_random_port ${rangeStart} ${rangeEnd}
# kill -9 $(lsof -t /dev/nvidia*)
unset LD_LIBRARY_PATH
source ~/anaconda3/etc/profile.d/conda.sh
source ~/Lib/anaconda3/etc/profile.d/conda.sh
source /file_system/vepfs/algorithm/ruiqi.song/anaconda3/etc/profile.d/conda.sh

mkdir -p $WORK_DIR
PERCENT=$2
# FOLD=$3
FOLD=5
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
GPUS=8

conda activate matesemidetr && python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port $PORT \
    tools/train_detr_ssod.py $1 \
    --launcher pytorch \
    --cfg-options fold=${FOLD} percent=${PERCENT} ${@:3}
