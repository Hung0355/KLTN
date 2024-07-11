import cv2
import mediapipe as mp
import numpy as np
import threading
import tensorflow as tf
import os
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import itertools

label = "Warmup...."
n_time_steps = 1
lm_list = []

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

model = tf.keras.models.load_model("model1.h5")

#cap = cv2.VideoCapture(0)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

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
        if img is not None:
            images.append(img)
        else:
            print(f"Failed to read image: {image_file}")

    return images

def make_landmark_timestep(results):
    c_lm = []
    for id, lm in enumerate(results.pose_landmarks.landmark):
        c_lm.append(lm.x)
        c_lm.append(lm.y)
        c_lm.append(lm.z)
        c_lm.append(lm.visibility)
    return c_lm


def draw_landmark_on_image(mpDraw, results, img):
    mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
    for id, lm in enumerate(results.pose_landmarks.landmark):
        h, w, c = img.shape
        #print(id, lm)
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
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


def detect(model, lm_list):
    global label
    lm_list = np.array(lm_list)
    lm_list = np.expand_dims(lm_list, axis=0)
    print(lm_list.shape)
    results = model.predict(lm_list)
    print("result:",results)
    if results[0][0] > 0.5:
        label = "NONE"
    else:
        label = "CHEAT"
    return label


cheat_path = 'D:/Anaconda/MiAI_Human_Activity_Recognition/test_data/cheat'
none_path = 'D:/Anaconda/MiAI_Human_Activity_Recognition/test_data/none'
cheat_images = read_images_from_folder(cheat_path)
cheat_labels = [1] * len(cheat_images)

none_images = read_images_from_folder(none_path)
none_labels = [0] * len(none_images)

# Concatenate images and labels
images = cheat_images + none_images
labels = cheat_labels + none_labels

detected_faces = []
index = 0
for image in images:
    index = index + 1
    img = image
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)

    if results.pose_landmarks:
        c_lm = make_landmark_timestep(results)

        lm_list.append(c_lm)
        if len(lm_list) == n_time_steps:
            print("image",index)
            # predict
            t1 = threading.Thread(target=detect, args=(model, lm_list,))
            t1.start()
            lm_list = []

        img = draw_landmark_on_image(mpDraw, results, img)

    img = draw_class_on_image(label, img)
    if label == 'CHEAT':
        detected_faces.append(1)
    else:
        detected_faces.append(0)
    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # if cv2.waitKey(1) == ord('q'):
    #     break

#cap.release()
cv2.destroyAllWindows()
accuracy = accuracy_score(labels, detected_faces)
print(f"Accuracy: {accuracy}")

# Precision
precision = precision_score(labels, detected_faces)
print(f"Precision: {precision}")

# Recall
recall = recall_score(labels, detected_faces)
print(f"Recall: {recall}")

# F1 Score
f1 = f1_score(labels, detected_faces)
print(f"F1 Score: {f1}")
cnf_matrix = confusion_matrix(labels, detected_faces)
class_names = [0, 1]
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                    title='Confusion matrix')
plt.show()
