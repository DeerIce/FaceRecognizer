import cv2
import os
import numpy as np
from sklearn import preprocessing
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox


def select_train_folder():
    selected_folder = filedialog.askdirectory()  # 使用askdirectory函数选择文件夹
    path_train.set(selected_folder)


def select_test_folder():
    selected_folder = filedialog.askdirectory()  # 使用askdirectory函数选择文件夹
    path_test.set(selected_folder)


def load_HaarCascade(path):
    # 载入Haar Cascade分类器
    face_cascade = cv2.CascadeClassifier(path)
    if face_cascade.empty():
        raise IOError('Unable to load cascade classifier xml file')
    else:
        return face_cascade


def train():
    if (not len(path_train.get())):
        messagebox.showerror('err: Path of train set is empty!')
        raise IOError('Unable to load cascade classifier xml file')

    print('Training...')
    train_status.set("Training...")

    labels_word = []
    images = []
    for root, dirs, files in os.walk(path_train.get()):
        for filename in (file for file in files if file.endswith('.jpg')):
            filepath = os.path.join(root, filename)
            image = cv2.imread(filepath, 0)
            faces1 = face_cascade.detectMultiScale(
                image, scaleFactor=1.3, minNeighbors=10)
            for (x, y, w, h) in faces1:
                images.append(image[y:y+h, x:x+w])
                labels_word.append(filepath.split('\\')[-2])
    labels_num = le.fit_transform(labels_word)

    recognizer.train(images, np.array(labels_num))
    train_status.set("Train")
    print('Train over!')


def predict():
    print('predicting...')
    stop_flag = False
    for root, dirs, files in os.walk(path_test.get()):
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
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 6)
                cv2.namedWindow('recognizing result', cv2.WINDOW_NORMAL)
                cv2.imshow('recognizing result', test_image)

            c = cv2.waitKey(0)
            if c == 27:
                stop_flag = True
                break
        if stop_flag:
            break
    print('predict over!')


if __name__ == '__main__':
    face_cascade = load_HaarCascade('haarcascade_frontalface_default.xml')
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    le = preprocessing.LabelEncoder()

    window = tk.Tk()
    window.title("Face Recognizer")
    window.geometry('450x300')
    window.config(background="#90C6FF")

    # 初始化Entry控件的textvariable属性值
    path_train = tk.StringVar()
    path_test = tk.StringVar()
    train_status = tk.StringVar()
    train_status.set("Train")

    # 布局控件
    tk.Label(window, width=20, text="Path of Training set：", bg="#90C6FF", anchor="center", fg="#ffffff").grid(
        column=0, row=0)
    tk.Entry(window, textvariable=path_train, width=30,bg="#afdfff").grid(
        column=1, row=0,ipady=3)
    tk.Button(window, text="select folder",
              command=select_train_folder).grid(row=0, column=2)

    tk.Label(window, width=20, text="Path of Testing set：", bg="#90C6FF", anchor="center", fg="#ffffff").grid(
        column=0, row=2)
    tk.Entry(window, textvariable=path_test, width=30,bg="#afdfff").grid(
        column=1, row=2,ipady=3)
    tk.Button(window, text="select folder",
              command=select_test_folder).grid(row=2, column=2)

    tk.Button(window, textvariable=train_status, command=lambda: train(),
              width=10, height=3).grid(row=4, column=0)
    tk.Button(window, text="predict", command=lambda: predict(),
              width=10, height=3).grid(row=4, column=1)

    window.mainloop()
