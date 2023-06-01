# import cv2
# import numpy as np
# import os
# import pickle

# # 加载人脸检测器和人脸识别模型
# face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# model = pickle.load(open('face_recognition_model.sav', 'rb'))

# # 加载人脸标签
# label_dict = {'0': 'Person 1', '1': 'Person 2', '2': 'Person 3'}  # 根据实际情况修改
# labels = list(label_dict.values())

# # 读取测试图片并进行人脸检测和人脸识别
# test_img = cv2.imread('test/3.jpg')
# gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
# faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
# for (x, y, w, h) in faces:
#     face_roi = gray[y:y+h, x:x+w]
#     face_roi = cv2.resize(face_roi, (224, 224))
#     face_roi = np.reshape(face_roi, (1, 224, 224, 1))
#     face_roi = face_roi / 255.0
#     preds = model.predict(face_roi)
#     label = labels[np.argmax(preds)]
#     cv2.rectangle(test_img, (x, y), (x+w, y+h), (255, 0, 0), 2)
#     cv2.putText(test_img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# cv2.imshow('Face Detection and Recognition', test_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

import os
import torch
import numpy as np
import cv2
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def get_embedding(img_path):
    img = Image.open(img_path)
    img_cropped = mtcnn(img)
    if img_cropped is None:
        return None
    img_embedding = resnet(img_cropped.unsqueeze(0).to(device))
    return img_embedding.detach().cpu().numpy()[0]

embeddings = []
for file_name in os.listdir('path/to/your/photos'):
    img_path = os.path.join('path/to/your/photos', file_name)
    img_embedding = get_embedding(img_path)
    if img_embedding is not None:
        embeddings.append(img_embedding)
embeddings = np.array(embeddings)
np.save('embeddings.npy', embeddings)

embeddings = np.load('embeddings.npy')
kmeans = KMeans(n_clusters=2, random_state=0).fit(embeddings)
labels = kmeans.labels_
le = LabelEncoder()
le.fit(labels)
np.save('labels.npy', le.transform(labels))
