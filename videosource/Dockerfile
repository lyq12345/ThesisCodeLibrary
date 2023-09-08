FROM benchpilot/raspbian-picamera2:latest
RUN apt-get update && apt-get -y upgrade
# make redis
RUN apt-get install -y gcc net-tools make
WORKDIR /usr/local/redis
ADD redis-6.2.1.tar.gz .
WORKDIR /usr/local/redis/redis-6.2.1/src
RUN make && make install
WORKDIR /usr/local/redis/redis-6.2.1
ADD redis.conf ./demo.conf
# make python and opencv
RUN apt-get install -y python3-opencv opencv-data
RUN pip install requests redis flask
# start services
ADD entrypoint.sh /
RUN chmod u+x /entrypoint.sh
# start flask
WORKDIR /app
COPY . .
#CMD [ "python3", "app2.py" ]
#WORKDIR /usr/local/redis/redis-6.2.1/src
#CMD ["./redis-server", "../demo.conf"]
ENTRYPOINT ["/entrypoint.sh"]
EXPOSE 6379
EXPOSE 5000