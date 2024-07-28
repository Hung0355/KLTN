1. HARDWARE
- COMPONENTS:
  
    ![image](https://github.com/user-attachments/assets/cb4fe7ed-3447-4360-a616-567fb316664f)
    - Esp32 Devkit v1
    - RFID RC522
    - LCD 1602
- FEATURES:
    - ESP32 read UID from RC522.
    - Send UID to server by MQTT.
    - Get student's seat from server.
    - Display name and seat on LCD.
2. FACE AND CHEATING ACTION REGCONITION
- Accuracy of facial recognition model : 94%
- Accuracy of cheating action recognition model: 99.61%
- FEATURES:
    - Read student's UID from MySQL.
    - Counting faces in a frame.
    - Recognize face.
    - Lock screen.
    - Dectect cheating action.
    - Display warning.
    - Send cheating images to MySQL.
 
3. GUI
- FRAMEWORK: PyQT6
  ![image](https://github.com/user-attachments/assets/a63ec827-c951-4662-aba7-3f9636eb9e24)

- FEATURES:
    - Upload students, classes, rooms lists to MySQL.
    - Delete table, table content.
    - Download cheating images.
    - Show table.
    - Create exam.

4.SERVER
- FRAMEWORK: FLASK
- FEATURES:
    - Get student's UID.
    - Read student's seat on MySQL.
    - Sent student's seat to ESP32 by MQTT.
  
