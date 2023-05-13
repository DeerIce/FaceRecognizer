import os
import cv2
from sklearn import preprocessing
import tkinter as tk

# le=preprocessing.LabelEncoder()
# label_words=[]
# for root, dirs, files in os.walk('train'):
#     for filename in (x for x in files if x.endswith('.jpg')):
#         filepath = os.path.join(root, filename)
#         # print(filepath)
#         label_words.append(filepath.split('\\')[-1])

# [print(i,end='\n') for i in label_words]
# LL=le.fit_transform(label_words)
# print(LL)
# LK=le.inverse_transform(LL)
# print(LK)


# def load_HaarCascade(path):
#     # 载入Haar Cascade分类器
#     face_cascade = cv2.CascadeClassifier(path)
#     if face_cascade.empty():
#         raise IOError('Unable to load cascade classifier xml file')
#     else:
#         return face_cascade

# images=[]
# face_cascade = load_HaarCascade('haarcascade_frontalface_default.xml')
# image = cv2.imread('test/GEM/t2.jpg', 0)
# print(image[1:50,1:50])
# faces = face_cascade.detectMultiScale(
#     image, scaleFactor=1.3, minNeighbors=10)
# # print('faces=',faces)
# for (x,y,w,h) in faces:
#     images.append(image[y:y+h,x:x+w])
#     cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
# # print("images=",images)
# cv2.imshow('img', image)
# cv2.waitKey()
# cv2.destroyAllWindows()


window = tk.Tk()
window.title("Face Recognizer")
window.geometry('450x300')

# 初始化Entry控件的textvariable属性值
path_train = tk.StringVar()
path_train.set('dasd')
print(path_train.get())