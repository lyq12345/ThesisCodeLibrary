import random
import time
import base64

from paho.mqtt import client as mqtt_client


broker = 'test.mosquitto.org'
port = 1883
topic = "/python/mqtt2"
# generate client ID with pub prefix randomly
client_id = f'python-mqtt-{random.randint(0, 1000)}'


def connect_mqtt():
    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print("Connected to MQTT Broker!")
        else:
            print("Failed to connect, return code %d\n", rc)

    client = mqtt_client.Client(client_id)
    client.on_connect = on_connect
    client.connect(broker, port)
    return client


def publish(client):
    msg_count = 0
    while True:
        with open("compressed.jpg", "rb") as file:
            filecontent = file.read()
            base64_bytes = base64.b64encode(filecontent)
            base64_message = base64_bytes.decode('ascii')
            time.sleep(5)
            result = client.publish(topic, base64_message)
            # result: [0, 1]
            status = result[0]
            if status == 0:
                print(f"Send message to topic `{topic}`")
            else:
                print(f"Failed to send message to topic {topic}")
            msg_count += 1


def run():
    client = connect_mqtt()
    client.loop_start()
    publish(client)


if __name__ == '__main__':
    run()