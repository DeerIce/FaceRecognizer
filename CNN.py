import os
import cv2
import numpy as np
import tensorflow as tf
from keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import load_model


def create_face_recognition_model():
    model = models.Sequential()
    # 使用修正线性单元(Rectified linear unit：relu）作为神经元的激活函数
    # Conv2D is a 2D Convolution Layer
    model.add(layers.Conv2D(
        filters=32,
        kernel_size=(5, 5),
        activation='relu',
        input_shape=(100, 100,3)))  # (96,96,32) @ 3
    '''每个滤波器（卷积核）都会与输入图像的每个通道进行卷积操作，生成一个单通道的特征图。
    有32个滤波器,则每个通道会有生成32个特征图:96x96
    图像有3个通道,所以总共会有 3 x 32 = 96 个特征图'''

    # MaxPooling2D is a 2D Pooling Layer
    model.add(layers.MaxPooling2D((2, 2)))  # (48,48,32) @ 3

    model.add(layers.Reshape((48, 48, 32, 1)))  # 调整输入形状为五维

    model.add(layers.Conv3D(
        filters=16,
        kernel_size=(5, 5, 32),
        activation='relu'))  # (44,44,1,16) @ 3

    model.add(layers.MaxPooling3D((2, 2, 1)))  # (22,22,1,16) @ 3
    model.add(layers.Reshape((22, 22, 16, 1)))

    model.add(layers.Conv3D(
        filters=16,
        kernel_size=(5, 5, 16),
        activation='relu'))  # (18,18,1,16) @ 3

    model.add(layers.MaxPooling3D((2, 2, 1)))  # (9,9,1,16) @ 3
    model.add(layers.Reshape((9, 9, 16, 1)))

    model.add(layers.Conv3D(
        filters=16,
        kernel_size=(5, 5, 16),
        activation='relu'))  # (5,5,1,16) @ 3

    model.add(layers.Reshape((5, 5, 16, 1)))
    model.add(layers.Conv3D(
        filters=128,
        kernel_size=(5, 5, 16),
        activation='relu'))  # (1,1,1,128) @ 3

    # 添加全局池化层
    model.add(layers.GlobalAveragePooling3D())

    # 添加全连接层
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(7, activation='softmax'))  # 根据具体的分类类别数量进行设置

    # # Flatten：把多维的输入一维化，常用在从卷积层到全连接层的过渡
    # model.add(layers.Flatten())
    # # Dense：全连接层
    # model.add(layers.Dense(128, activation='relu'))
    # model.add(layers.Dense(128, activation='relu'))
    # model.add(layers.Dense(7, activation='softmax'))

    #编译模型
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def load_data_from_folders(folder_path):
    images = []
    labels = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.jpg'):
                img_path = os.path.join(root, file)
                img = cv2.imread(img_path)
                img = cv2.resize(img, (100, 100))
                img = img.astype('float32') / 255.0
                images.append(img)
                labels.append(os.path.basename(root))
    return np.array(images), np.array(labels)


def train_model(model, train_data, train_labels, test_data, test_labels):
    model.fit(train_data, train_labels, epochs=10,
              validation_data=(test_data, test_labels))


def predict_person(model, image_path, label_encoder):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (100, 100))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    predicted_label = np.argmax(prediction, axis=1)
    return label_encoder.inverse_transform(predicted_label)


if __name__ == '__main__':
    images, labels = load_data_from_folders('train')
    # print(labels)

    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    # 随机划分训练集和测试集
    train_data, test_data, train_labels, test_labels = train_test_split(
        images, encoded_labels, test_size=0.2, random_state=42)

    print(type(train_data))
    print(type(test_data))
    print(type(train_labels))
    print(type(test_labels))

    # model = create_face_recognition_model()
    # train_model(model, train_data, train_labels, test_data, test_labels)

    # # model.save('model1.h5')

    # new_image_path = 'test/00020.jpg'
    # person_name = predict_person(model, new_image_path, label_encoder)
    # print(f'The person in the image is: {person_name}')
