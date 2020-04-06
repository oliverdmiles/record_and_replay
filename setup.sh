#!/bin/bash
set -Eeuo pipefail
sudo bash -c 'echo 0 > /proc/sys/kernel/randomize_va_space'
