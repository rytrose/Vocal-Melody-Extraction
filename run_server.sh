#!/usr/bin/env bash

source activate shimi
python server.py &
ssh -R shimi-dataset-server:80:localhost:5000 serveo.net
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT