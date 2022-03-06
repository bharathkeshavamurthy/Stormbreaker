#!/bin/bash

declare -a strsToSend=('Hello ...' 'My name is Python ...' 'What is your name?' '...')

for s in ${strsToSend[@]}
do
    py -2.7 ./ModifiedPyFiles/Test2_baseband_rx_file_sink_Mod.py
done