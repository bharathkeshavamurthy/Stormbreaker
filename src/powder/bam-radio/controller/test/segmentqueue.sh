#!/bin/bash

# https://stackoverflow.com/questions/2734144/linux-disable-using-loopback-and-send-data-via-wire-between-2-eth-cards-of-one
# https://serverfault.com/questions/127636/force-local-ip-traffic-to-an-external-interface

[ $# -eq 1 ] || exit 1

pkill bamradio
#pkill nc

mytest="$(pwd)/$1"
echo "$mytest"

OUTDIR="sqtest.$(date +%s)"
mkdir $OUTDIR
cd $OUTDIR

#echo 1 | sudo tee /proc/sys/net/ipv4/ip_no_pmtu_disc
apparmor_parser -R /etc/apparmor.d/usr.sbin.tcpdump
ulimit -c unlimited

$mytest >| test.stdout 2>&1 &
sleep 1
tcpdump -i tun0 -w tun0.pcap &
tcpdump -i tun1 -w tun1.pcap &

sleep 10
pkill bamradio
pkill tcpdump
exit

iptables -F -v
iptables -F -v -t nat

IP0=172.16.241.1
IP00=172.16.241.100
IP1=172.16.242.1
IP11=172.16.242.100
IP2=172.16.243.1
IP22=172.16.243.100
IP3=172.16.244.1
IP33=172.16.244.100
IP4=172.16.245.1
IP44=172.16.245.100
IP5=172.16.246.1
IP55=172.16.246.100

ip route add $IP00 dev tun1
ip route add $IP11 dev tun0
ip route add $IP22 dev tun0
ip route add $IP33 dev tun0
ip route add $IP44 dev tun0
ip route add $IP55 dev tun0
# change source address
iptables -t nat -A POSTROUTING -s $IP0 -j SNAT --to-source $IP00
iptables -t nat -A POSTROUTING -s $IP1 -j SNAT --to-source $IP11
iptables -t nat -A POSTROUTING -s $IP2 -j SNAT --to-source $IP22
iptables -t nat -A POSTROUTING -s $IP3 -j SNAT --to-source $IP33
iptables -t nat -A POSTROUTING -s $IP4 -j SNAT --to-source $IP44
iptables -t nat -A POSTROUTING -s $IP5 -j SNAT --to-source $IP55
# change dest address
iptables -t nat -A PREROUTING -d $IP00 -j DNAT --to-destination $IP0
iptables -t nat -A PREROUTING -d $IP11 -j DNAT --to-destination $IP1
iptables -t nat -A PREROUTING -d $IP22 -j DNAT --to-destination $IP2
iptables -t nat -A PREROUTING -d $IP33 -j DNAT --to-destination $IP3
iptables -t nat -A PREROUTING -d $IP44 -j DNAT --to-destination $IP4
iptables -t nat -A PREROUTING -d $IP55 -j DNAT --to-destination $IP5


openssl rand -out txfile 1000000
nc -v -l $IP0 1100 >| nc1100.out &
nc -v -l $IP0 1101 >| nc1101.out &
nc -v -l $IP0 1102 >| nc1102.out &
nc -v -l $IP0 1103 >| nc1103.out &
nc -v -l $IP0 1104 >| nc1104.out &
nc -v -l $IP0 1105 >| nc1105.out &
nc -v -l $IP1 1200 >| nc1200.out &
nc -v -l $IP1 1201 >| nc1201.out &
nc -v -l $IP1 1202 >| nc1202.out &
nc -v -l $IP1 1203 >| nc1203.out &
nc -v -l $IP1 1204 >| nc1204.out &
nc -v -l $IP1 1205 >| nc1205.out &
sleep 1
nc -v -s $IP1 $IP00 1100 < txfile &
nc -v -s $IP1 $IP00 1101 < txfile &
nc -v -s $IP1 $IP00 1102 < txfile &
nc -v -s $IP1 $IP00 1103 < txfile &
nc -v -s $IP1 $IP00 1104 < txfile &
nc -v -s $IP1 $IP00 1105 < txfile &
nc -v -s $IP0 $IP11 1200 < txfile &
nc -v -s $IP0 $IP11 1201 < txfile &
nc -v -s $IP0 $IP11 1202 < txfile &
nc -v -s $IP0 $IP11 1203 < txfile &
nc -v -s $IP0 $IP11 1204 < txfile &
nc -v -s $IP0 $IP11 1205 < txfile &

sleep 5
echo "listing jobs"
jobs
#for ncnum in $(seq 1 24); do echo "wait nc $ncnum"; wait %; done
echo sleeping
sleep 15
pkill bamradio-segmentqueue-tests
