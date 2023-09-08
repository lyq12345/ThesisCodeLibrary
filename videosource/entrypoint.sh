#!/bin/sh

# start redis server
cd /usr/local/redis/redis-6.2.1/src
./redis-server ../demo.conf &

# return to root dir
cd /app
python3 app2.py