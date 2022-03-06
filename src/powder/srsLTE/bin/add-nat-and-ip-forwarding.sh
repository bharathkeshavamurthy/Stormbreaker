#!/bin/bash
sudo /sbin/iptables -t nat -A POSTROUTING -o `cat /var/emulab/boot/controlif` -j MASQUERADE
sudo /sbin/sysctl -w net.ipv4.ip_forward=1
