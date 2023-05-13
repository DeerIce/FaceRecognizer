import cv2
import os
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn import preprocessing


def load_HaarCascade(path):
    # 载入Haar Cascade分类器
    face_cascade = cv2.CascadeClassifier(path)
    if face_cascade.empty():
        raise IOError('Unable to load cascade classifier xml file')
    else:
        return face_cascade


if __name__ == '__main__':
    path_cascade = 'haarcascade_frontalface_default.xml'
    path_train = 'Python-Machine-Learning-Cookbook-master/Chapter10/faces_dataset/train'
    path_test = 'Python-Machine-Learning-Cookbook-master/Chapter10/faces_dataset/test'

    face_cascade = load_HaarCascade(path_cascade)

    '''Train'''
    le = preprocessing.LabelEncoder()
    labels_word = []
    images = []
    for root, dirs, files in os.walk(path_train):
        for filename in (file for file in files if file.endswith('.jpg')):
            filepath = os.path.join(root, filename)
            image = cv2.imread(filepath, 0)
            faces1 = face_cascade.detectMultiScale(
                image, scaleFactor=1.3, minNeighbors=10)
            for (x, y, w, h) in faces1:
                images.append(image[y:y+h, x:x+w])
                labels_word.append(filepath.split('\\')[-2])
    labels_num = le.fit_transform(labels_word)

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(images, np.array(labels_num))

    ''' Test'''
    stop_flag = False
    for root, dirs, files in os.walk(path_test):
        for filename in (file for file in files if file.endswith('.jpg')):
            filepath = os.path.join(root, filename)
            test_image = cv2.imread(filepath, 0)
            # 检测人脸. scaleFactor: 每个阶段的比例系数
            faces2 = face_cascade.detectMultiScale(
                test_image, scaleFactor=1.3, minNeighbors=10)
            for (x, y, w, h) in faces2:
                predicted_label, conf = recognizer.predict(
                    test_image[y:y+h, x:x+w])
                predicted_name = le.inverse_transform([predicted_label])[0]
                cv2.putText(test_image, predicted_name, (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 10, 50), 6)
                cv2.namedWindow('recognizing result', cv2.WINDOW_NORMAL)
                cv2.imshow('recognizing result', test_image)

            c = cv2.waitKey(0)
            if c == 27:
                stop_flag = True
                break
        if stop_flag:
            break
