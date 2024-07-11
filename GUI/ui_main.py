import sys
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLineEdit, QComboBox, QFileDialog, QLabel, QTableWidget, QTableWidgetItem
)
from PyQt6.QtCore import Qt
import mysql.connector
import pandas as pd
import os
import random

class ExamApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Quản lý môn thi và thí sinh")
        self.setGeometry(100, 100, 800, 600)
        
        # Kết nối đến MySQL
        self.connection = mysql.connector.connect(
            host="103.241.43.129",
            user="hung",  # Thay bằng username của bạn
            password="0355",  # Thay bằng password của bạn
            database="exam",  # Thay bằng tên database của bạn
            ssl_disabled=True,  # Disable SSL
            use_pure=True
        )
        self.cursor = self.connection.cursor()
        
        self.initUI()
        
    def initUI(self):
        centralWidget = QWidget()
        self.setCentralWidget(centralWidget)
        
        mainLayout = QVBoxLayout()
        
        self.examInput = QLineEdit(self)
        self.examInput.setPlaceholderText("Choose Rooms/Students/Exams")
        self.addExamButton = QPushButton("Tạo bảng")
        self.addExamButton.clicked.connect(self.addExam)
        
        examLayout = QHBoxLayout()
        examLayout.addWidget(self.examInput)
        examLayout.addWidget(self.addExamButton)
        
        self.examComboBox = QComboBox(self)
        self.updateExamComboBox()
        
        self.fileButton = QPushButton("Upload file excel")
        self.fileButton.clicked.connect(self.loadFile)
        
        self.displayButton = QPushButton("Hiển thị thông tin")
        self.displayButton.clicked.connect(self.displayStudents)

        self.createButton = QPushButton("Tạo kỳ thi")
        self.createButton.clicked.connect(self.createExam)
        
        self.deleteButton = QPushButton("Xóa bảng")
        self.deleteButton.clicked.connect(self.deleteExam)

        self.clearButton = QPushButton("Xóa dữ liệu bảng")
        self.clearButton.clicked.connect(self.clearExamData)

        self.downloadButton = QPushButton("Tải hình ảnh cheating")
        self.downloadButton.clicked.connect(self.DownloadIMG)

        self.reloadButton = QPushButton("Reload")
        self.reloadButton.clicked.connect(self.reloadUI)
        
        fileLayout = QHBoxLayout()
        fileLayout.addWidget(self.examComboBox)
        fileLayout.addWidget(self.fileButton)
        fileLayout.addWidget(self.displayButton)
        fileLayout.addWidget(self.createButton)
        fileLayout.addWidget(self.deleteButton)
        fileLayout.addWidget(self.clearButton)
        fileLayout.addWidget(self.downloadButton)
        fileLayout.addWidget(self.reloadButton)
        
        self.tableWidget = QTableWidget()
        
        mainLayout.addLayout(examLayout)
        mainLayout.addLayout(fileLayout)
        mainLayout.addWidget(self.tableWidget)
        
        centralWidget.setLayout(mainLayout)
    
    def addExam(self):
        examName = self.examInput.text()
        if examName == 'Rooms':
            createTableQuery = f"""
            CREATE TABLE IF NOT EXISTS `{examName}` (
                code varchar(20) not null primary key,
                numseat tinyint(255)
            );
            """
        elif examName == 'Students':
            createTableQuery = f"""
            CREATE TABLE IF NOT EXISTS `{examName}` (
                UID varchar(20) not null,
                ClassCode varchar(255) not null,
                Name varchar(255),
                PRIMARY KEY(UID, ClassCode),
                FOREIGN KEY (ClassCode) REFERENCES Exams(ClassCode)
            );
            """
        elif examName == 'Exams':
            createTableQuery = f"""
            CREATE TABLE IF NOT EXISTS `{examName}` (
                ClassCode varchar(255),
                subject varchar(255),
                date DATE,
                period int,
                code varchar(20),
                primary key (ClassCode),
                foreign key (code) references Rooms(code)
            );
            """
        else:
            return None
        self.cursor.execute(createTableQuery)
        self.connection.commit()
        self.updateExamComboBox()
        self.examInput.clear()
    
    def updateExamComboBox(self):
        self.examComboBox.clear()
        self.cursor.execute("SHOW TABLES;")
        exams = [row[0] for row in self.cursor.fetchall()]
        self.examComboBox.addItems(exams)
    
    def loadFile(self):
        filePath, _ = QFileDialog.getOpenFileName(self, "Chọn file Excel", "", "Excel Files (*.xlsx)")
        if filePath:
            selectedExam = self.examComboBox.currentText()
            if selectedExam == 'Rooms':
                self.loadRoom(filePath)
            elif selectedExam == 'Students':
                print("Start")
                self.loadStudent(filePath)
            elif selectedExam == 'Exams':
                self.loadExam(filePath)
            self.updateTable(selectedExam)

    def loadRoom(self,filePath):
        if filePath:
            df = pd.read_excel(filePath)
            for index, row in df.iterrows():
                selectQuery = f"SELECT * FROM Rooms WHERE code = %s"
                self.cursor.execute(selectQuery, (row['CODE'],))
                result = self.cursor.fetchone()
                if result:
                    updateQuery = f"UPDATE Rooms SET numseat = %s WHERE code = %s"
                    self.cursor.execute(updateQuery, (row['NUMSEAT'], row['CODE']))
                else:
                    insertQuery = f"INSERT INTO Rooms (code, numseat) VALUES (%s, %s)"
                    self.cursor.execute(insertQuery, (row['CODE'], row['NUMSEAT']))
            self.connection.commit()

    def loadStudent(self,filePath):
        if filePath:
            df = pd.read_excel(filePath)
            for index, row in df.iterrows():
                selectQuery = f"SELECT * FROM Students WHERE UID = %s AND ClassCode = %s"
                self.cursor.execute(selectQuery, (row['UID'],row['ClassCode']))
                result = self.cursor.fetchone()
                if result:
                    updateQuery = f"UPDATE Students SET Name = %s WHERE UID = %s AND ClassCode = %s"
                    self.cursor.execute(updateQuery, (row['Name'],row['UID'],row['ClassCode']))
                else:
                    insertQuery = f"INSERT INTO Students (UID, ClassCode,Name) VALUES (%s, %s,%s)"
                    self.cursor.execute(insertQuery, (row['UID'],row['ClassCode'],row['Name']))
            self.connection.commit()

    def loadExam(self,filePath):
        if filePath:
            df = pd.read_excel(filePath)
            df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y').dt.strftime('%Y-%m-%d')
            for index, row in df.iterrows():
                selectQuery = f"SELECT * FROM Exams WHERE ClassCode = %s"
                self.cursor.execute(selectQuery, (row['ClassCode'],))
                result = self.cursor.fetchone()
                if result:
                    updateQuery = f"UPDATE Exams SET subject = %s, date = %s, period = %s, code = %s WHERE ClassCode = %s"
                    self.cursor.execute(updateQuery, (row['subj'], row['date'], row['period'], row['room'], row['ClassCode']))
                else:
                    insertQuery = f"INSERT INTO Exams (ClassCode, subject, date, period, code) VALUES (%s, %s, %s, %s, %s)"
                    self.cursor.execute(insertQuery, (row['ClassCode'], row['subj'], row['date'], row['period'], row['room']))
            self.connection.commit()            
    
    def updateTable(self, examName):
        self.cursor.execute(f"SELECT * FROM `{examName}`")
        rows = self.cursor.fetchall()
        columns = [i[0] for i in self.cursor.description]
        
        self.tableWidget.setColumnCount(len(columns))
        self.tableWidget.setRowCount(len(rows))
        self.tableWidget.setHorizontalHeaderLabels(columns)
        
        for rowIdx, row in enumerate(rows):
            for colIdx, item in enumerate(row):
                self.tableWidget.setItem(rowIdx, colIdx, QTableWidgetItem(str(item)))

    def displayStudents(self):
        selectedExam = self.examComboBox.currentText()
        if selectedExam:
            self.updateTable(selectedExam)
    
    def createExam(self):
        createTableQuery = f"""
            CREATE TABLE IF NOT EXISTS ExamRooms (
                UID VARCHAR(20),
                Name VARCHAR(255),
                ClassCode VARCHAR(255),
                date DATE,
                period INT,
                code VARCHAR(20),
                seat INT,
                PRIMARY KEY (code, seat, period),
                FOREIGN KEY (UID, ClassCode) REFERENCES Students(UID, ClassCode),
                FOREIGN KEY (ClassCode) REFERENCES Exams(ClassCode),
                FOREIGN KEY (code) REFERENCES Rooms(code)
            );
            """
        self.cursor.execute(createTableQuery)
        createQuery = f"""
            SELECT s.UID, s.Name, e.ClassCode, e.date, e.period, r.code, r.numseat 
            FROM Students s
            JOIN Exams e ON s.ClassCode = e.ClassCode
            JOIN Rooms r ON e.code = r.code
            ORDER BY r.code, e.period, e.ClassCode
        """        
        self.cursor.execute(createQuery)
        exam_data = self.cursor.fetchall()
        used_seats = {}
        cur_period = 0
        for (UID, Name, ClassCode, date, period, code, numseat) in exam_data:
            if (code not in used_seats) or (cur_period != period):
                cur_period = period
                used_seats[code] = list(range(1, numseat + 1))
                print(used_seats)
            seat = random.choice(used_seats[code])
            used_seats[code].remove(seat)

            self.cursor.execute("""
                INSERT INTO ExamRooms (UID, Name, ClassCode, date, period, code, seat) 
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (UID, Name, ClassCode, date, period, code, seat))
        print(used_seats)
        self.connection.commit()
    
    def deleteExam(self):
        selectedExam = self.examComboBox.currentText()
        if selectedExam:
            deleteQuery = f"DROP TABLE `{selectedExam}`"
            self.cursor.execute(deleteQuery)
            self.connection.commit()
            self.updateExamComboBox()
            self.tableWidget.clear()
            self.tableWidget.setRowCount(0)
            self.tableWidget.setColumnCount(0)

    def clearExamData(self):
        selectedExam = self.examComboBox.currentText()
        if selectedExam:
            clearQuery = f"DELETE FROM `{selectedExam}`"
            self.cursor.execute(clearQuery)
            self.connection.commit()
            self.updateTable(selectedExam)

    def DownloadIMG(self):
        Query = "SELECT * FROM cheating"
        self.cursor.execute(Query)
        results = self.cursor.fetchall()
        for row in results:
            image_id = row[0]
            name = row[1]
            time = row[3]
            image_data = row[4]
            time_str = str(time).replace(":", "_")

            directory = os.path.join('cheating_image', name)
            if not os.path.exists(directory):
                os.makedirs(directory)

            # Tên tệp hình ảnh
            filename = os.path.join(directory, f'{image_id}_{time_str}.jpg')
            # Ghi dữ liệu hình ảnh vào tệp
            with open(filename, 'wb') as file:
                file.write(image_data)
            #print(f'Hình ảnh với ID {image_id} đã được tải xuống và lưu dưới tên {filename}.')
    
    def reloadUI(self):
        self.updateExamComboBox()
        selectedExam = self.examComboBox.currentText()
        if selectedExam:
            self.updateTable(selectedExam)
                
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ExamApp()
    window.show()
    sys.exit(app.exec())


# CREATE TABLE cheating (
#     id INT AUTO_INCREMENT PRIMARY KEY,
#     name varchar(20),
#     uid varchar(20),
#     time TIME,
#     image_data LONGBLOB
# );
