FROM benchpilot/raspbian-picamera2:latest
RUN apt update && apt-get -y upgrade
# make redis
RUN apt install -y gcc net-tools make
WORKDIR /usr/local/redis
ADD redis-6.2.1.tar.gz .
WORKDIR /usr/local/redis/redis-6.2.1/src
RUN make && make install
WORKDIR /usr/local/redis/redis-6.2.1
ADD redis.conf ./demo.conf
WORKDIR /usr/local/redis/redis-6.2.1/src
CMD ["./redis-server", "../demo.conf"]