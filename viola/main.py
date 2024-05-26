import cv2 as cv
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools

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




# Load the pre-trained face cascade classifier
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')


face_path = []
noneface_path = []

for i in range(1,51):
    face_path.append(r'D:/Anaconda/viola/data/face/Human'+str(i)+'.png')
for i in range(10,60):
    noneface_path.append(r'D:/Anaconda/viola/data/noneface/00'+str(i)+'.png')


# Load images and create labels
# Assume you have two lists: face_images and nonface_images, each containing the paths to face and non-face images
# Create labels for face_images: 1 for face, 0 for non-face
face_images = [cv.imread(path) for path in face_path]
face_labels = [1] * len(face_path)
# Create labels for nonface_images: 0 for non-face, 1 for face
noneface_images = [cv.imread(path) for path in noneface_path]
nonface_labels = [0] * len(noneface_images)

# Concatenate images and labels
images = face_images + noneface_images
labels = face_labels + nonface_labels

# Convert images to grayscale and detect faces
detected_faces = []
for image in images:
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_image)
    if len(faces) > 0:
        detected_faces.append(1)  # Face detected
    else:
        detected_faces.append(0)  # No face detected


# Convert detected_faces to numpy array
detected_faces = np.array(detected_faces)


# Plot confusion matrix

cnf_matrix = confusion_matrix(labels, detected_faces)
class_names = [0, 1]
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix')
plt.show()


