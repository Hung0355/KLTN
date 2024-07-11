import cv2
import mediapipe as mp
import pandas as pd
import os

# Đọc ảnh từ webcam
#cap = cv2.VideoCapture(0)

# Khởi tạo thư viện mediapipe
mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

lm_list = []
none_label = "NONE"
cheat_label = "CHEATING"

none_label1 = "NONE_TEST"
cheat_label1 = "CHEATING_TEST"

no_of_frames = 200

def read_images_from_folder(folder_path):
    # List to store the images
    images = []

    # Supported image file extensions
    supported_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

    # Get list of all files in the folder
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(supported_extensions)]

    # Iterate over the files and read images
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        img = cv2.imread(image_path)
        #img = cv2.resize(img, (1200, 1000))  # Example size, adjust as needed
        if img is not None:
            images.append(img)
        else:
            print(f"Failed to read image: {image_file}")

    return images

def make_landmark_timestep(results):
    print(results.pose_landmarks.landmark)
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

# Train data
cheat_path = 'D:/Anaconda/MiAI_Human_Activity_Recognition/trained_data/pack1/cheat'
none_path = 'D:/Anaconda/MiAI_Human_Activity_Recognition/trained_data/pack1/none'

# Test data
cheat_path1 = 'D:/Anaconda/MiAI_Human_Activity_Recognition/test_data/pack1/cheat'
none_path1 = 'D:/Anaconda/MiAI_Human_Activity_Recognition/test_data/pack1/none'

cheat_images = read_images_from_folder(cheat_path1)
#trained_labels = [1] * len(cheat_images)

none_images = read_images_from_folder(none_path1)
#none_labels = [0] * len(none_images)

for image in none_images:
    frame = image
    # Nhận diện pose
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frameRGB)

    if results.pose_landmarks:
        # Ghi nhận thông số khung xương
        lm = make_landmark_timestep(results)
        lm_list.append(lm)
        # Vẽ khung xương lên ảnh
        frame = draw_landmark_on_image(mpDraw, results, frame)

    cv2.imshow("image", frame)
    #cv2.waitKey(0)
    if cv2.waitKey(1) == ord('q'):
        break

# Write vào file csv
df  = pd.DataFrame(lm_list)
df.to_csv(none_label1 + ".txt")
#cap.release()
cv2.destroyAllWindows()