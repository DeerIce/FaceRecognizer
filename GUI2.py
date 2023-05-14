import cv2
import os
import re
import copy
from PIL import Image, ImageTk
import numpy as np
from sklearn import preprocessing
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import threading


class GUI():
    def __init__(self) -> None:
        self.cascade()

        self.init_Window()
        self.init_StringVar()
        self.train_folder_msg()
        self.test_folder_msg()
        self.button_train_predict()

        self.window.mainloop()

    def cascade(self):
        self.face_cascade = load_HaarCascade(
            'haarcascade_frontalface_default.xml')
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.le = preprocessing.LabelEncoder()

    def init_StringVar(self):
        self.path_train = tk.StringVar()
        self.path_test = tk.StringVar()
        self.train_status = tk.StringVar()  # 用于在按钮上显示train/training
        self.train_status.set("Train")
        self.train_over = False
        self.test_images = []
        self.predicted_names = []

    def init_Window(self):
        self.window = tk.Tk()
        self.window.title("Face Recognizer")
        self.window.geometry('485x300')
        self.window.config(background="#90C6FF")

    def train_folder_msg(self):
        tk.Label(self.window, width=17, text="Path of Training set：", bg="#90C6FF", anchor="center", fg="#ffffff").grid(
            column=0, row=0)
        tk.Entry(self.window, textvariable=self.path_train, width=39, bg="#afdfff").grid(
            column=1, row=0, ipady=3)
        tk.Button(self.window, text="select folder",
                  command=self.select_train_folder).grid(row=0, column=2)

    def test_folder_msg(self):
        tk.Label(self.window, width=17, text="Path of Testing set：", bg="#90C6FF", anchor="center", fg="#ffffff").grid(
            column=0, row=2)
        tk.Entry(self.window, textvariable=self.path_test, width=39, bg="#afdfff").grid(
            column=1, row=2, ipady=3)
        tk.Button(self.window, text="select folder",
                  command=self.select_test_folder).grid(row=2, column=2)

    def select_train_folder(self):
        selected_folder = filedialog.askdirectory()  # 使用askdirectory函数选择文件夹
        self.path_train.set(selected_folder)
        self.train_over = False

    def select_test_folder(self):
        selected_folder = filedialog.askdirectory()  # 使用askdirectory函数选择文件夹
        self.path_test.set(selected_folder)

    def button_train_predict(self):
        tk.Button(self.window, textvariable=self.train_status, command=self.train,
                  width=10, height=3).grid(row=4, column=0)
        tk.Button(self.window, text="Predict", command=self.predict,
                  width=10, height=3).grid(row=4, column=1)
        tk.Button(self.window, text="currentThread",
                  command=self.print_activeCount).grid(row=4, column=2)

    def __train(self):
        if (not len(self.path_train.get())):
            messagebox.showerror('err: Path of train set is empty!')
            raise IOError('Unable to load cascade classifier xml file')

        print('Training...')
        self.train_status.set("Training...")

        labels_word = []
        images = []
        for root, dirs, files in os.walk(self.path_train.get()):
            # print("root="+root)
            for filename in (file for file in files if file.endswith('.jpg')):
                filepath = os.path.join(root, filename)
                image = cv2.imread(filepath, 0)
                faces1 = self.face_cascade.detectMultiScale(
                    image, scaleFactor=1.3, minNeighbors=10)
                for (x, y, w, h) in faces1:
                    images.append(image[y:y+h, x:x+w])
                    labels_word.append(re.split(r'[/\\]', filepath)[-2])
                    # print(re.split(r'[/\\]',filepath))
        labels_num = self.le.fit_transform(labels_word)

        self.recognizer.train(images, np.array(labels_num))
        self.train_status.set("Train")
        self.train_over = True
        print('Train over!')

    def train(self):
        train_thread = threading.Thread(target=self.__train)
        train_thread.start()

    def __predict(self):
        if (self.train_over):
            print('predicting...')
            stop_flag = False
            for root, dirs, files in os.walk(self.path_test.get()):
                for filename in (file for file in files if file.endswith('.jpg')):
                    filepath = os.path.join(root, filename)
                    test_image = cv2.imread(filepath, 0)
                    # 检测人脸. scaleFactor: 每个阶段的比例系数
                    faces2 = self.face_cascade.detectMultiScale(
                        test_image, scaleFactor=1.3, minNeighbors=10)
                    for (x, y, w, h) in faces2:
                        predicted_label, conf = self.recognizer.predict(
                            test_image[y:y+h, x:x+w])
                        predicted_name = self.le.inverse_transform(
                            [predicted_label])[0]
                        self.predicted_names.append(predicted_name)
                        self.test_images.append(test_image)

                #         cv2.putText(test_image, predicted_name, (10, 60),
                #                     cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 6)
                #         cv2.namedWindow('recognizing result',
                #                         cv2.WINDOW_NORMAL)
                #         cv2.imshow('recognizing result', test_image)

                #     c = cv2.waitKey(0)
                #     if c == 27:
                #         stop_flag = True
                #         break
                # if stop_flag:
                #     break
            print('predict over!')
            # print('face_length=',len(self.test_images))
            # print('name_length=',len(self.predicted_names))
            self.show_result()
        else:
            messagebox.showerror('err: Have not train!')

    def predict(self):
        predict_thread = threading.Thread(target=self.__predict)
        predict_thread.start()

    def print_activeCount(self):
        print(threading.activeCount())

    def __show_result(self):
        resultWindow = ResultWindow(self.test_images, self.predicted_names)

    def show_result(self):
        result_thread = threading.Thread(target=self.__show_result)
        result_thread.start()


class ResultWindow():
    def __init__(self, test_images, predicted_names):
        self.images = copy.deepcopy(test_images)
        self.names = copy.deepcopy(predicted_names)
        self.index = 0
        self.w_box, self.h_box = 350, 350

        self.init_Window()
        self.image_cv2tk()
        self.layout()

    def init_Window(self):
        self.window = tk.Toplevel()
        self.window.title("Predict Result")
        self.window.geometry('500x500')

    def layout(self):
        if (len(self.images) > 0):
            self.img_label = tk.Label(
                self.window, image=self.photos[self.index])
            # self.img_label.grid(row=0, column=0)
            self.img_label.place(x=0, y=0)

            self.name_label = tk.Label(
                self.window, text=self.names[self.index], anchor="center",
                bg="#b7f5fb", font=("微软雅黑", 36))
            # self.name_label.grid(row=1, column=0)
            self.name_label.place(x=300, y=150)

        tk.Button(self.window, text='Next',
                  command=self.button_change_image, width=15, height=1).place(x=170, y=400)

    def button_change_image(self):
        self.index += 1
        if (self.index > len(self.images)-1):
            self.index = 0
        self.img_label.configure(image=self.photos[self.index])
        self.name_label.configure(text=self.names[self.index])

    def image_cv2tk(self):
        self.pil_imgs = [Image.fromarray(cv2.cvtColor(
            image, cv2.COLOR_GRAY2RGB)) for image in self.images]
        self.image_resize()
        self.photos = [ImageTk.PhotoImage(pil_img)
                       for pil_img in self.pil_imgs]

    def image_resize(self):
        resized_pil_imgs = []
        for pil_img in self.pil_imgs:
            w, h = pil_img.size
            k = min([1.0*self.w_box/w, 1.0*self.h_box/h])
            w_new = int(w*k)
            h_new = int(h*k)
            resized_pil_imgs.append(pil_img.resize((w_new, h_new)))
        self.pil_imgs = copy.deepcopy(resized_pil_imgs)


def load_HaarCascade(path):
    # 载入Haar Cascade分类器
    face_cascade = cv2.CascadeClassifier(path)
    if face_cascade.empty():
        raise IOError('Unable to load cascade classifier xml file')
    else:
        return face_cascade


if __name__ == '__main__':
    myGUI = GUI()
