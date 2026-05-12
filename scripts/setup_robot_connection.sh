#!/usr/bin/env bash
# Bring up the wired link to the Unitree Go2 and verify reachability.
#
# Set IFACE to your robot-facing Ethernet device (find it with `ip link`).
# The Go2's default IP is 192.168.123.161; we put the host at .99/24.

set -euo pipefail

IFACE="${IFACE:-enx98fc84e68f1a}"
ROBOT_IP="${ROBOT_IP:-192.168.123.161}"
HOST_IP="${HOST_IP:-192.168.123.99/24}"

echo "Bringing up $IFACE -> $HOST_IP"
sudo ip addr flush dev "$IFACE"
sudo ip addr add "$HOST_IP" dev "$IFACE"
sudo ip link set "$IFACE" up

echo "Pinging robot at $ROBOT_IP"
if ! ping -c 3 "$ROBOT_IP"; then
    echo "Ping failed. Scanning 192.168.123.0/24 to locate the robot..."
    sudo nmap -sn 192.168.123.0/24
    exit 1
fi

echo "Robot reachable. Activate the deployment env with:  conda activate env_go2"
