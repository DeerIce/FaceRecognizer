import cv2
import os
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn import preprocessing


class label_encoder():
    def __init__(self):
        self.le = preprocessing.LabelEncoder()

    def encode_labels(self, labels):
        self.le.fit(labels)

    def word2num(self, label):
        pass


def load_HaarCascade(path):
    # 载入Haar Cascade分类器
    face_cascade = cv2.CascadeClassifier(path)
    if face_cascade.empty():
        raise IOError('Unable to load cascade classifier xml file')
    else:
        return face_cascade


def draw_rectangle(img_path):
    # 载入Haar Cascade分类器
    face_cascade = load_HaarCascade()
    # 读入图片
    img = cv2.imread(img_path)
    # 转为灰度图像
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 检测人脸. scaleFactor: 每个阶段的比例系数
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.3, minNeighbors=5)
    # 标出人脸框
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    # 显示结果
    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    cv2.imshow('img', img)
    cv2.waitKey()
    cv2.destroyAllWindows()


def get_images_and_labels(path):
    le = preprocessing.LabelEncoder()
    labels_word = []
    images = []
    for root, dirs, files in os.walk(path):
        for filename in (file for file in files if file.endswith('.jpg')):
            filepath = os.path.join(root, filename)
            images.append(cv2.imread(filepath, 0))
            labels_word.append(filepath.split('\\')[-2])
    labels_num = le.fit_transform(labels_word)
    return images, labels_num, le


if __name__ == '__main__':
    path_cascade = 'haarcascade_frontalface_default.xml'
    path_train = 'train'
    path_test = 'test'

    '''Train'''
    images, labels_num, le = get_images_and_labels(path_train)
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(images, np.array(labels_num))

    ''' Test'''
    stop_flag = False
    face_cascade = load_HaarCascade(path_cascade)
    for root, dirs, files in os.walk(path_test):
        for filename in (file for file in files if file.endswith('.jpg')):
            filepath = os.path.join(root, filename)
            test_image = cv2.imread(filepath, 0)
            # 检测人脸. scaleFactor: 每个阶段的比例系数
            faces = face_cascade.detectMultiScale(
                test_image, scaleFactor=1.3, minNeighbors=5)
            for (x, y, w, h) in faces:
                predicted_label, conf = recognizer.predict(
                    test_image[y:y+h, x:x+w])
                predicted_name = le.inverse_transform([predicted_label])[0]
                cv2.putText(test_image, predicted_name, (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 6)
                cv2.imshow('recognizing result',test_image)

            c = cv2.waitKey(0)
            if c == 27:
                stop_flag = True
                break





# if __name__ == '__main__':
#     '''人脸检测'''
#     draw_rectangle('test.jpg')

#     '''人脸识别'''
#     # 1.特征提取： PCA
#     # 1.1 载入Haar Cascade分类器
#     face_cascade = load_HaarCascade()

#     # 1.2 读入人脸数据集
#     data = np.load('face_data.npy', allow_pickle=True).item()
#     X_train = data['X_train']
#     y_train = data['y_train']

#     # 1.3 将训练数据转换为二维矩阵
#     X_train_flat = X_train.reshape(X_train.shape[0], -1)

#     # 1.4 计算PCA特征
#     pca = cv2.PCA(n_components=80)
#     pca.fit(X_train_flat)

#     # 1.5 载入测试图片
#     img = cv2.imread('test.jpg')
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#     # 1.6 检测人脸
#     faces = face_cascade.detectMultiScale(
#         gray, scaleFactor=1.3, minNeighbors=5)

#     # 1.7 对每个人脸进行识别
#     for (x, y, w, h) in faces:
#         # 对人脸进行预处理
#         face_img = gray[y:y+h, x:x+w]
#         face_img_resized = cv2.resize(face_img, (100, 100))
#         face_img_flat = face_img_resized.reshape(1, -1)

#         # 计算PCA特征
#         face_pca = pca.transform(face_img_flat)

#         # 预测人脸的ID
#         face_id = -1
#         min_dist = float('inf')
#         for i in range(len(X_train)):
#             dist = np.linalg.norm(
#                 face_pca - pca.transform(X_train[i].reshape(1, -1)))
#             if dist < min_dist:
#                 min_dist = dist
#                 face_id = y_train[i]

#         # 在图像上显示预测结果
#         cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
#         cv2.putText(img, str(face_id), (x, y-10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

#     # 显示结果
#     cv2.imshow('img', img)
#     cv2.waitKey()
#     cv2.destroyAllWindows()

#     '''人脸识别'''
#     # 2.特征提取： SVM
#     # 读入训练数据
#     data = np.load('gender_data.npz')
#     X_train = data['X_train']
#     y_train = data['y_train']

#     # 将训练数据转换为一维矩阵
#     X_train_flat = X_train.reshape(X_train.shape[0], -1)

#     # 训练SVM模型
#     svm = SVC(kernel='linear', C=10, gamma=0.001)
#     svm.fit(X_train_flat, y_train)

#     # 读入测试图片
#     img = cv2.imread('test.jpg')
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#     # 检测人脸
#     faces = face_cascade.detectMultiScale(
#         gray, scaleFactor=1.2, minNeighbors=5)

#     # 对每个人脸进行识别
#     for (x, y, w, h) in faces:
#         # 对人脸进行预处理
#         face_img = gray[y:y+h, x:x+w]
#         face_img_resized = cv2.resize(face_img, (100, 100))
#         face_img_flat = face_img_resized.reshape(1, -1)

#         # 预测人脸的性别
#         gender_id = svm.predict(face_img_flat)

#         # 在图像上显示预测结果
#         cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
#         cv2.putText(img, 'Male' if gender_id == 1 else 'Female',
#                     (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

#     # 显示结果
#     cv2.imshow('img', img)
#     cv2.waitKey()
#     cv2.destroyAllWindows()
