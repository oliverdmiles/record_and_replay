#!/bin/bash
set -Eeuo pipefail

export LD_PRELOAD=/home/omiles/582/record_and_replay/record_and_replay.so
export CUDA_VISIBLE_DEVICES=1
sudo bash -c 'echo 0 > /proc/sys/kernel/randomize_va_space'



