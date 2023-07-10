from flask import Flask, jsonify
from pijuice import PiJuice

app = Flask(__name__)
pijuice = PiJuice(1, 0x14)  # create PiJuice object

@app.route('/battery')
def get_battery():
    battery_status = pijuice.status.GetStatus()['data']  # Get battery status
    return jsonify({
        'battery_percentage': battery_status['battery'],
        'battery_voltage': battery_status['batteryVoltage']
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
