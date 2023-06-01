import cv2
import numpy as np

# 加载人脸检测器模型
prototxt_path = "deploy.prototxt"
weights_path = "res10_300x300_ssd_iter_140000.caffemodel"
net = cv2.dnn.readNetFromCaffe(prototxt_path, weights_path)

def extract_face_features(image):
    # 构建输入图像的blob
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    
    # 通过神经网络进行人脸检测
    net.setInput(blob)
    detections = net.forward()
    
    # 提取检测到的人脸
    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])
            (x, y, w, h) = box.astype("int")
            face = image[y:y+h, x:x+w]
            face = cv2.resize(face, (100, 100))  # 调整为相同的大小，方便比较
            faces.append(face)
    
    return faces

def compare_faces(image1, image2):
    # 提取两张图片中的人脸特征
    faces1 = extract_face_features(image1)
    faces2 = extract_face_features(image2)
    
    if len(faces1) == 0 or len(faces2) == 0:
        # 图片中未检测到人脸
        return False
    
    # 计算欧氏距离
    distances = []
    for face1 in faces1:
        for face2 in faces2:
            distance = np.linalg.norm(face1 - face2)
            distances.append(distance)
    
    # 设置距离阈值
    threshold = 30000
    print(distances)
    
    if min(distances) < threshold:
        return True  # 是同一个人
    else:
        return False  # 不是同一个人

# 示例用法
image1_path = "test/00019.jpg"
image2_path = "test/Abid_Hamid_Mahmud_Al-Tikriti_0003.jpg"
image1 = cv2.imread(image1_path)
image2 = cv2.imread(image2_path)
result = compare_faces(image1, image2)
print("两张图片中的人是否是同一个人：", result)
