import paho.mqtt.publish as publish
MQTT_SERVER = "169.234.25.41"  #Write Server IP Address
MQTT_PATH = "image/"

f=open("test.jpg", "rb") #3.7kiB in same folder
fileContent = f.read()
byteArr = bytearray(fileContent)


publish.single(MQTT_PATH, byteArr, hostname=MQTT_SERVER)
