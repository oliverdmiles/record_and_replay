#!/bin/bash
#
# run_record_replay
#
# Run a record and replay job automatically

# Stop on errors, print commands
# See https://vaneyckt.io/posts/safer_bash_scripts_with_set_euxo_pipefail/
set -Eeuo pipefail

usage() {
    echo 'USAGE: ./run.sh EXEC'
    echo '       EXEC          path to the executable to record and replay'
    echo 'RUN setup.sh to disable ASLR if you care'
}

if [[ "$#" -ne 1 ]]; then
    echo "Illegal number of parameters"
    usage
    exit 1
fi


CURRENT_DIR="$( pwd )"
EXECUTABLE=$1
FULL_EXEC=$CURRENT_DIR/$EXECUTABLE

export CUDA_VISIBLE_DEVICES=1
export LD_PRELOAD=$CURRENT_DIR/output/record_and_replay.so
RECORD_REPLAY_PHASE=0 $FULL_EXEC

export LD_PRELOAD=
echo "This is where I would process the data"

export LD_PRELOAD=$CURRENT_DIR/output/record_and_replay.so
RECORD_REPLAY_PHASE=1 $FULL_EXEC
