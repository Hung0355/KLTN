from flask import Flask
from flask_mqtt import Mqtt
from datetime import datetime
import mysql.connector
import os

app = Flask(__name__)

app.config['MQTT_BROKER_URL'] = 'mqtt.flespi.io'
app.config['MQTT_BROKER_PORT'] = 1883 
app.config['MQTT_USERNAME'] = 'FlespiToken yKpDALXhIKxS2HU1iPHd2Xg28UheeZn16T9ohsBTRxTgJO97mlmL7zXUa9gTO8Fa'
app.config['MQTT_PASSWORD'] = '*'
app.config['MQTT_REFRESH_TIME'] = 1.0 # refresh time in seconds

mqtt = Mqtt(app)

@mqtt.on_connect()
def handle_connect(client, userdata, flags, rc):
    mqtt.subscribe('nodeEsp32')
    if rc == 0:
        print("connected")
    else:
        print("connect failed")
@mqtt.on_message()
def handle_mqtt_message(client, userdata, message):
    data = dict(
        topic=message.topic,
        payload=message.payload.decode()
    )
    print(data['payload'])
    if data['topic'] == 'nodeEsp32':
        #doan nay lay data trong database thong qua bien data['payload'] la id, lay duoc name trong database thi bo vo ham mqtt.publish('send_rfid','name')
        cnx = mysql.connector.connect(
            host="103.241.43.129",
            user="hung",  # Thay bằng username của bạn
            password="0355",  # Thay bằng password của bạn
            database="exam",  # Thay bằng tên database của bạn
            ssl_disabled=True,  # Disable SSL
            use_pure=True
        )
        cursor = cnx.cursor()
        id = data['payload']
        current_time = datetime.now().time()
        current_date = datetime.now().strftime("%Y-%m-%d")

        print(current_date)
        # In ra giờ, phút, giây
        if datetime.strptime('07:30', '%H:%M').time() <= current_time <= datetime.strptime('08:20', '%H:%M').time():
            period = 1
        elif datetime.strptime('09:00', '%H:%M').time() <= current_time <= datetime.strptime('08:45', '%H:%M').time():
            period = 2
        elif datetime.strptime('13:00', '%H:%M').time() <= current_time <= datetime.strptime('13:45', '%H:%M').time():
            period = 3
        elif datetime.strptime('14:30', '%H:%M').time() <= current_time <= datetime.strptime('15:15', '%H:%M').time():
            period = 4
        else:
            period = 4  # Nếu không thuộc vào bất kỳ khoảng thời gian nào
        # Thực hiện truy vấn
        query = "SELECT seat, Name FROM ExamRooms WHERE UID = %s AND period = %s AND date = %s"
        cursor.execute(query, (id,period,current_date))
        seat = cursor.fetchone()
        #print(seat)
        if seat:
            result = str(seat[0]) + '/' + seat[1]
            print(result)

        ########################################
    if seat:
        mqtt.publish('nodeServer',result)
    else:
        mqtt.publish('nodeServer',data['payload'])

if __name__ == '__main__':
    print("hello")
    while(1):
        None