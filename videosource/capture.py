from picamera2 import Picamera2
import yaml
import time
import os

cur_path = os.path.dirname(os.path.realpath(__file__))
config_path = os.path.join(cur_path, "capture_config.yml")
f = open(config_path, 'r', encoding='utf-8')
cfg = f.read()
config_dict = yaml.load(cfg, Loader=yaml.FullLoader)
# print(config_dict)

# camera setup and preview
picam2 = Picamera2()
camera_config = picam2.create_preview_configuration()
picam2.configure(camera_config)

raw_path = '/var/tmp/raw.jpg'
compressed_path = '/var/tmp/compressed.jpg'

try:
    # set capture interval
    capture_interval = 5 if 'interval' not in config_dict else float(config_dict['interval'])  # capture one image every 5 seconds
    picam2.start()
    while True:
        # begin capturing

        # wait for sometime
        time.sleep(capture_interval)

        # capture one image
        picam2.capture_file(raw_path)
        print("image captured!")

except KeyboardInterrupt:
    picam2.stop()
    print("capture stopped")

finally:
    # turn off camera
    picam2.stop()
