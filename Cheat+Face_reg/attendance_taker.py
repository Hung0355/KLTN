import dlib
import numpy as np
import cv2
import os
import pandas as pd
import time
import logging
import mediapipe as mp
import threading
import tensorflow as tf
import mysql.connector
import wx
import ctypes
from datetime import datetime

ROOM = 'A101'
SEAT = '22'
prev_uid = None
prev_name = None
# Dlib  / Use frontal face detector of Dlib
detector = dlib.get_frontal_face_detector()

# Dlib landmark / Get face landmarks
predictor = dlib.shape_predictor('data/data_dlib/shape_predictor_68_face_landmarks.dat')

# Dlib Resnet Use Dlib resnet50 model to get 128D face descriptor
face_reco_model = dlib.face_recognition_model_v1("data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")

# Create a connection to the database
# conn = sqlite3.connect("attendance.db")
# cursor = conn.cursor()

# # Create a table for the current date
# current_date = datetime.datetime.now().strftime("%Y_%m_%d")  # Replace hyphens with underscores
# table_name = "attendance" 
# create_table_sql = f"CREATE TABLE IF NOT EXISTS {table_name} (name TEXT, time TEXT, date DATE, UNIQUE(name, date))"
# cursor.execute(create_table_sql)


# # Commit changes and close the connection
# conn.commit()
# conn.close()

label = "Waiting...."
n_time_steps = 10

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

model = tf.keras.models.load_model("model.h5")

cnx = mysql.connector.connect(
    host='103.241.43.129',    # Địa chỉ máy chủ MySQL
    user='hung', # Tên đăng nhập
    password='0355', # Mật khẩu
    database='exam', # Tên cơ sở dữ liệu
    ssl_disabled=True,  # Disable SSL
    use_pure=True
    )
    # Tạo một đối tượng cursor
cursor = cnx.cursor()
def Get_Mysql(room,seat):
    current_time = datetime.now().time()
    current_date = datetime.now().strftime("%Y-%m-%d")
    print("date:",current_date)
    print("time:",current_time)
    if datetime.strptime('07:30', '%H:%M').time() <= current_time <= datetime.strptime('08:15', '%H:%M').time():
        period = 1
    elif datetime.strptime('09:00', '%H:%M').time() <= current_time <= datetime.strptime('08:45', '%H:%M').time():
        period = 2
    elif datetime.strptime('13:00', '%H:%M').time() <= current_time <= datetime.strptime('13:45', '%H:%M').time():
        period = 3
    elif datetime.strptime('14:30', '%H:%M').time() <= current_time <= datetime.strptime('15:30', '%H:%M').time():
        period = 4
    else:
        period = 4  # Nếu không thuộc vào bất kỳ khoảng thời gian nào
    # Thực hiện truy vấn
    query = f"SELECT UID FROM ExamRooms WHERE seat = %s AND period = %s AND code = %s AND date = %s"
    cursor.execute(query,(seat,period,room,current_date))
    uid = cursor.fetchone()
    
    # query = f"SELECT name FROM `{room}` WHERE uid = %s"
    # cursor.execute(query,(uid[0],))
    #nam = cursor.fetchone()

    return uid[0]

def Get_NameMysql(uid):
    cursor1 = cnx.cursor()
    global prev_uid
    global prev_name
    if uid == prev_uid:
        return prev_name
    prev_uid = uid
    if uid == 'unknown':
        return uid
    # Thực hiện truy vấn
    query = f"SELECT Name FROM ExamRooms WHERE UID = %s"
    cursor1.execute(query,(uid,))
    name = cursor1.fetchone()
    print('Name tim thay: ',name[0])
    if not name:
        return 'unknown'
    prev_name = name[0]
    return name[0]

def Capture(img):
    query = "SELECT Name FROM ExamRooms WHERE UID = %s"
    cursor.execute(query, (uid,))
    Name_mysql = cursor.fetchone()
    cursor.fetchall()  # Đảm bảo rằng tất cả kết quả truy vấn được xử lý
    if Name_mysql:
        cur_time = datetime.now().strftime('%H:%M:%S')
        _, buffer = cv2.imencode('.jpg', img)
        image_data = buffer.tobytes()
        print(Name_mysql[0])
        query = "INSERT INTO cheating (name, uid, time,image_data) VALUES (%s, %s, %s, %s)"
        cursor.execute(query, (Name_mysql[0], uid, cur_time, image_data))
        cnx.commit()

user32 = ctypes.windll.user32

def lock_screen():
    user32.LockWorkStation()

def show_message():
    app = wx.App(False)
    wx.MessageBox('Cheating detect!', 'Warning', wx.OK | wx.ICON_INFORMATION)

def make_landmark_timestep(results):
    c_lm = []
    face_landmarks = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    for id in face_landmarks:
        lm = results.pose_landmarks.landmark[id]
        c_lm.append(lm.x)
        c_lm.append(lm.y)
        c_lm.append(lm.z)
        c_lm.append(lm.visibility)
    return c_lm


def draw_landmark_on_image(mpDraw, results, img):
    # Vẽ các đường nối
    mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)

    # Vẽ các điểm nút của khuôn mặt
    face_landmarks = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    for id in face_landmarks:
        lm = results.pose_landmarks.landmark[id]
        h, w, c = img.shape
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(img, (cx, cy), 10, (0, 0, 255), cv2.FILLED)
    return img

def draw_class_on_image(label, img):
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10, 30)
    fontScale = 1
    fontColor = (0, 255, 0)
    thickness = 2
    lineType = 2
    cv2.putText(img, label,
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                thickness,
                lineType)
    return img

def detect(model, lm_list,img):
    global label
    lm_list = np.array(lm_list)
    lm_list = np.expand_dims(lm_list, axis=0)
    print(lm_list.shape)
    results = model.predict(lm_list)
    print(results)
    if results[0][0] > 0.5:
        label = "NONE"
    else:
        label = "CHEAT"
        show_message()
        Capture(img)
    return label

class Face_Recognizer:
    def __init__(self):
        self.font = cv2.FONT_ITALIC
        # FPS
        self.frame_time = 0
        self.frame_start_time = 0
        self.fps = 0
        self.fps_show = 0
        self.start_time = time.time()

        # cnt for frame
        self.frame_cnt = 0

        #  Save the features of faces in the database
        self.face_features_known_list = []
        # / Save the name of faces in the database
        self.face_name_known_list = []

        #  List to save centroid positions of ROI in frame N-1 and N
        self.last_frame_face_centroid_list = []
        self.current_frame_face_centroid_list = []

        # List to save names of objects in frame N-1 and N
        self.last_frame_face_name_list = []
        self.current_frame_face_name_list = []

        #  cnt for faces in frame N-1 and N
        self.last_frame_face_cnt = 0
        self.current_frame_face_cnt = 0

        # Save the e-distance for faceX when recognizing
        self.current_frame_face_X_e_distance_list = []

        # Save the positions and names of current faces captured
        self.current_frame_face_position_list = []
        #  Save the features of people in current frame
        self.current_frame_face_feature_list = []

        # e distance between centroid of ROI in last and current frame
        self.last_current_frame_centroid_e_distance = 0

        #  Reclassify after 'reclassify_interval' frames
        self.reclassify_interval_cnt = 0
        self.reclassify_interval = 10

    #  "features_all.csv"  / Get known faces from "features_all.csv"
    def get_face_database(self):
        if os.path.exists("data/features_all.csv"):
            path_features_known_csv = "data/features_all.csv"
            csv_rd = pd.read_csv(path_features_known_csv, header=None)
            for i in range(csv_rd.shape[0]):
                features_someone_arr = []
                self.face_name_known_list.append(csv_rd.iloc[i][0])
                for j in range(1, 129):
                    if csv_rd.iloc[i][j] == '':
                        features_someone_arr.append('0')
                    else:
                        features_someone_arr.append(csv_rd.iloc[i][j])
                self.face_features_known_list.append(features_someone_arr)
            logging.info("Faces in Database： %d", len(self.face_features_known_list))
            return 1
        else:
            logging.warning("'features_all.csv' not found!")
            logging.warning("Please run 'get_faces_from_camera.py' "
                            "and 'features_extraction_to_csv.py' before 'face_reco_from_camera.py'")
            return 0

    def update_fps(self):
        now = time.time()
        # Refresh fps per second
        if str(self.start_time).split(".")[0] != str(now).split(".")[0]:
            self.fps_show = self.fps
        self.start_time = now
        self.frame_time = now - self.frame_start_time
        self.fps = 1.0 / self.frame_time
        self.frame_start_time = now

    @staticmethod
    # / Compute the e-distance between two 128D features
    def return_euclidean_distance(feature_1, feature_2):
        feature_1 = np.array(feature_1)
        feature_2 = np.array(feature_2)
        dist = np.sqrt(np.sum(np.square(feature_1 - feature_2)))
        return dist

    # / Use centroid tracker to link face_x in current frame with person_x in last frame
    def centroid_tracker(self):
        for i in range(len(self.current_frame_face_centroid_list)):
            e_distance_current_frame_person_x_list = []
            #  For object 1 in current_frame, compute e-distance with object 1/2/3/4/... in last frame
            for j in range(len(self.last_frame_face_centroid_list)):
                self.last_current_frame_centroid_e_distance = self.return_euclidean_distance(
                    self.current_frame_face_centroid_list[i], self.last_frame_face_centroid_list[j])

                e_distance_current_frame_person_x_list.append(
                    self.last_current_frame_centroid_e_distance)

            last_frame_num = e_distance_current_frame_person_x_list.index(
                min(e_distance_current_frame_person_x_list))
            self.current_frame_face_name_list[i] = self.last_frame_face_name_list[last_frame_num]

    #  Face detection and recognition wit OT from input video stream
    def process1(self, stream):
        # 1.  Get faces known from "features.all.csv"
        lm_list = []
        key = 100
        global uid
        nam = 0
        count_unknown = 0
        count_name = 10
        uid = Get_Mysql(ROOM,SEAT)
        if self.get_face_database():
            while stream.isOpened():
                self.frame_cnt += 1
                logging.debug("Frame " + str(self.frame_cnt) + " starts")
                flag, img_rd = stream.read()
                kk = cv2.waitKey(1)

                # 2.  Detect faces for frame X
                faces = detector(img_rd, 0)

                # 3.  Update cnt for faces in frames
                self.last_frame_face_cnt = self.current_frame_face_cnt
                self.current_frame_face_cnt = len(faces)

                # 4.  Update the face name list in last frame
                self.last_frame_face_name_list = self.current_frame_face_name_list[:]

                # 5.  update frame centroid list
                self.last_frame_face_centroid_list = self.current_frame_face_centroid_list
                self.current_frame_face_centroid_list = []

                # 6.1  if cnt not changes
                if (self.current_frame_face_cnt == self.last_frame_face_cnt) and (
                        self.reclassify_interval_cnt != self.reclassify_interval):
                    logging.debug("scene 1:   No face cnt changes in this frame!!!")

                    self.current_frame_face_position_list = []

                    if "unknown" in self.current_frame_face_name_list:
                        self.reclassify_interval_cnt += 1

                    if self.current_frame_face_cnt != 0:
                        for k, d in enumerate(faces):
                            self.current_frame_face_position_list.append(tuple(
                                [faces[k].left(), int(faces[k].bottom() + (faces[k].bottom() - faces[k].top()) / 4)]))
                            self.current_frame_face_centroid_list.append(
                                [int(faces[k].left() + faces[k].right()) / 2,
                                 int(faces[k].top() + faces[k].bottom()) / 2])

                            # img_rd = cv2.rectangle(img_rd,
                            #                        tuple([d.left(), d.top()]),
                            #                        tuple([d.right(), d.bottom()]),
                            #                        (255, 255, 255), 2)

                    #  Multi-faces in current frame, use centroid-tracker to track
                    if self.current_frame_face_cnt != 1:
                        self.centroid_tracker()

                    for i in range(self.current_frame_face_cnt):
                        # 6.2 Write names under ROI
                        img_rd = cv2.putText(img_rd, self.current_frame_face_name_list[i],
                                             self.current_frame_face_position_list[i], self.font, 0.8, (0, 255, 255), 1,
                                             cv2.LINE_AA)
                    #self.draw_note(img_rd)
                    if key == 1:
                        imgRGB = cv2.cvtColor(img_rd, cv2.COLOR_BGR2RGB)
                        results = pose.process(imgRGB)
                        if results.pose_landmarks:
                            c_lm = make_landmark_timestep(results)
                            lm_list.append(c_lm)
                        if len(lm_list) == n_time_steps:
                            # predict
                            t1 = threading.Thread(target=detect, args=(model, lm_list,img_rd,))
                            t1.start()
                            lm_list = []

                        #img_rd = draw_landmark_on_image(mpDraw, results, img_rd)

                        img_rd = draw_class_on_image(label, img_rd)


                # 6.2  If cnt of faces changes, 0->1 or 1->0 or ...
                else:
                    nam = 0
                    logging.debug("scene 2: / Faces cnt changes in this frame")
                    self.current_frame_face_position_list = []
                    self.current_frame_face_X_e_distance_list = []
                    self.current_frame_face_feature_list = []
                    self.reclassify_interval_cnt = 0

                    # 6.2.1  Face cnt decreases: 1->0, 2->1, ...
                    if self.current_frame_face_cnt == 0:
                        logging.debug("  / No faces in this frame!!!")
                        # clear list of names and features
                        self.current_frame_face_name_list = []
                        Capture(img_rd)
                    # 6.2.2 / Face cnt increase: 0->1, 0->2, ..., 1->2, ...
                    else:
                        if (len(faces)>1):
                            key = 0
                            Capture(img_rd)
                            #lock_screen()
                        logging.debug("  scene 2.2  Get faces in this frame and do face recognition")
                        self.current_frame_face_name_list = []
                        for i in range(len(faces)):
                            shape = predictor(img_rd, faces[i])
                            self.current_frame_face_feature_list.append(
                                face_reco_model.compute_face_descriptor(img_rd, shape))
                            self.current_frame_face_name_list.append("unknown")

                        # 6.2.2.1 Traversal all the faces in the database
                        for k in range(len(faces)):
                            logging.debug("  For face %d in current frame:", k + 1)
                            self.current_frame_face_centroid_list.append(
                                [int(faces[k].left() + faces[k].right()) / 2,
                                 int(faces[k].top() + faces[k].bottom()) / 2])

                            self.current_frame_face_X_e_distance_list = []

                            # 6.2.2.2  Positions of faces captured
                            self.current_frame_face_position_list.append(tuple(
                                [faces[k].left(), int(faces[k].bottom() + (faces[k].bottom() - faces[k].top()) / 4)]))

                            # 6.2.2.3 
                            # For every faces detected, compare the faces in the database
                            for i in range(len(self.face_features_known_list)):
                                # 
                                if str(self.face_features_known_list[i][0]) != '0.0':
                                    e_distance_tmp = self.return_euclidean_distance(
                                        self.current_frame_face_feature_list[k],
                                        self.face_features_known_list[i])
                                    logging.debug("      with person %d, the e-distance: %f", i + 1, e_distance_tmp)
                                    self.current_frame_face_X_e_distance_list.append(e_distance_tmp)
                                else:
                                    #  person_X
                                    self.current_frame_face_X_e_distance_list.append(999999999)

                            # 6.2.2.4 / Find the one with minimum e distance
                            similar_person_num = self.current_frame_face_X_e_distance_list.index(
                                min(self.current_frame_face_X_e_distance_list))

                            if min(self.current_frame_face_X_e_distance_list) < 0.3:
                                count_unknown = 0
                                self.current_frame_face_name_list[k] = self.face_name_known_list[similar_person_num]
                                logging.debug("  Face recognition result: %s",
                                              self.face_name_known_list[similar_person_num])
                                
                                # Insert attendance record
                                nam =self.face_name_known_list[similar_person_num]

                                print(type(self.face_name_known_list[similar_person_num]))
                                print("uid:",uid)
                                if nam == uid:
                                    key = 1
                                    count_unknown = 0
                                else:
                                    key = 0
                                    lock_screen()
                                    lm_list = []
                                    Capture(img_rd)
                                #self.attendance(nam)
                            else:
                                nam = 'unknown'
                                key = 0
                                logging.debug("  Face recognition result: Unknown person")
                                count_unknown = count_unknown + 1
                                if (count_unknown >= 10):
                                    count_unknown = 0
                                    lock_screen()
                                    lm_list = []

                        # 7.  / Add note on cv2 window
                        #self.draw_note(img_rd)

                # 8.  'q'  / Press 'q' to exit
                if kk == ord('q'):
                    break

                self.update_fps()
                cv2.namedWindow("camera", 1)
                cv2.imshow("camera", img_rd)

                logging.debug("Frame ends\n\n")

    


    def run(self):
        # cap = cv2.VideoCapture("video.mp4")  # Get video stream from video file
        cap = cv2.VideoCapture(0)              # Get video stream from camera
        self.process1(cap)

        cap.release()
        cv2.destroyAllWindows()
    
   


def main():
    # logging.basicConfig(level=logging.DEBUG) # Set log level to 'logging.DEBUG' to print debug info of every frame
    logging.basicConfig(level=logging.INFO)
    Face_Recognizer_con = Face_Recognizer()
    Face_Recognizer_con.run()


if __name__ == '__main__':
    main()