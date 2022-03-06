#!/bin/bash

# core dump
ulimit -c unlimited

SRNID="$(hostname | grep -oE "[0-9]{1,3}$")"

COLO_CONF_PATH="/root/radio_api/colosseum_config.ini"
RADIO_CONF_PATH="/root/radio.conf"
if ! [[ -f $RADIO_CONF_PATH ]]; then
    echo "File not found: $RADIO_CONF_PATH"
    exit 1
fi
COLLAB_LOG_PATH="/logs/collab_client.log"

# copy colosseum_config
cp $COLO_CONF_PATH /logs/

# copy radio.conf
cp $RADIO_CONF_PATH /logs/

if [[ -d "/sys/class/net/col0" ]]; then
	COLLABNET="$(ip addr show col0 | grep "inet\b" | awk '{print $2}' | cut -d . -f3)"
	/root/bamradio "--config=$RADIO_CONF_PATH" "--colosseum_config=$COLO_CONF_PATH" "--phy_control.id=$SRNID" "--collaboration.gateway" "--collaboration.netid=$COLLABNET" "--collaboration.log_filename=$COLLAB_LOG_PATH" "--global.batch" &>> "/logs/node_gateway_stdout.txt"
else
	/root/bamradio "--config=$RADIO_CONF_PATH" "--colosseum_config=$COLO_CONF_PATH" "--phy_control.id=$SRNID" "--global.batch" &>> "/logs/node_stdout.txt"
fi
