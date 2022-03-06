#!/bin/bash

ulimit -c unlimited

# Always ask SRN ID in interactive mode
echo "Insert the SRN ID followed by [Enter]"
read SRNID

read -r -p "Gateway node? [y/N] " response

COLLAB_LOG_PATH="/logs/collab_client.log"

if [[ "$response" =~ ^([yY][eE][sS]|[yY])+$ ]]; then
	COLLABNET="$(ip addr show col0 | grep "inet\b" | awk '{print $2}' | cut -d . -f3)"
	COLO_CONF_PATH="/root/radio_api/colosseum_config.ini"
	/root/bamradio "--config=/root/radio_api/radio_default.conf" "--phy_control.id=$SRNID" "--colosseum_config=$COLO_CONF_PATH" "--collaboration.gateway" "--collaboration.netid=$COLLABNET" "--collaboration.log_filename=$COLLAB_LOG_PATH"
else
	/root/bamradio "--config=/root/radio_api/radio_default.conf" "--phy_control.id=$SRNID"
fi
