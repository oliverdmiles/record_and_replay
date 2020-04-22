#!/bin/bash
#
# run_record_replay
#
# Run a record and replay job automatically

# Stop on errors, print commands
# See https://vaneyckt.io/posts/safer_bash_scripts_with_set_euxo_pipefail/
set -Eeuo pipefail

usage() {
    echo 'USAGE: ./run.sh EXEC [OPTIONS] '
    echo ''
    echo 'EXEC                     path to the executable to record and replay'
    echo 'Options:'
    echo '       --record          only run the record phase'
    echo '       --filter          only run the data cleaning phase'
    echo '       --replay          only run the replay phase'
    echo '       --help            display usage message'
    echo ''
    echo 'RUN setup.sh to disable ASLR if you care'
}

record() {
    rm -rf record_output
    mkdir record_output
    export LD_PRELOAD=$CURRENT_DIR/output/record_and_replay.so
    RECORD_REPLAY_PHASE=0 NOBANNER=1 $FULL_EXEC
}

filter() {
    export LD_PRELOAD=
    rm -rf dependency_output
    mkdir dependency_output
    $CURRENT_DIR/detector $CURRENT_DIR/record_output/*
}

replay() {
    export LD_PRELOAD=$CURRENT_DIR/output/record_and_replay.so
    RECORD_REPLAY_PHASE=1 NOBANNER=1 $FULL_EXEC
}

if [[ "$#" -eq 1 || "$#" -eq 2 ]]; then
    export CUDA_VISIBLE_DEVICES=1
    CURRENT_DIR="$( pwd )"
    EXECUTABLE=$1
    FULL_EXEC=$CURRENT_DIR/$EXECUTABLE

    if [[ "$#" -eq 1 ]]; then
        if [[ "$1" == "--help" ]]; then
            usage 
            exit 0
        fi
        record
        filter
        replay
    else
        case "$2" in
            '--record')
            record
            ;;
            '--filter')
            filter
            ;;
            '--replay')
            replay
            ;;
            '--help')
            usage
            exit 0
            ;;
            *)
            echo "Invalid option"
            usage
            exit 1
            ;;
            esac
    fi
else
    echo "Illegal number of parameters"
    usage
    exit 1
fi