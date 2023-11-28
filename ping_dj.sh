#!/bin/bash

ip_address="134.76.19.44"

while true
do
    if ping -c 4 "$ip_address"; then
        echo "Ping to $ip_address successful!"
    else
        echo "Ping to $ip_address failed!"
    fi

    sleep 900  # 900 seconds = 15 minutes
done