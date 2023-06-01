import os
import cv2
import numpy as np
import tensorflow as tf
from keras import layers
from sklearn.preprocessing import LabelEncoder
from PIL import Image


def test_one_hot():
    indices = [1, 5, 2]  # 输入数据(是个向量),不能是字符串
    depth = 3
    print(tf.one_hot(indices, depth))  # output: [3 x 3]
    '''
        tf.Tensor(
        [[0. 1. 0.]
        [0. 0. 0.]
        [0. 0. 1.]], shape=(3, 3), dtype=float32)
    '''


def creat_model():
    # 创建一个顺序模型
    model = tf.keras.Sequential()
    # 添加第一个卷积层
    model.add(layers.Conv2D(
        32, (3, 3), activation="relu", input_shape=(100, 100, 1)))
    model.add(layers.MaxPooling2D((2, 2)))  # -> 13x13x32
    # 添加第二个卷积层
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))  # -> 11x11x64
    model.add(layers.MaxPooling2D((2, 2)))
    # 添加第三个卷积层
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    # 将三维特征转换为一维特征向量
    model.add(layers.Flatten())
    # 添加第一个全连接层
    model.add(layers.Dense(64, activation="relu"))
    # 添加第二个全连接层
    model.add(layers.Dense(7, activation="softmax"))

    return model


def load_data():
    # 加载MNIST数据集
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    # print('x_train: \n', x_train)
    # print('y_train: \n', y_train)
    # print('y_train:\n', y_train)
    # img = Image.fromarray(x_train[0])
    # img.show()
    # print(len(y_train))
    print(y_test)
    print(len(y_test))
    print(len(y_test[1]))

    # 对图像数据进行归一化和重新调整大小
    x_train = x_train.astype("float32") / 255.0  # astype用来转换数组的数据类型
    x_test = x_test.astype("float32") / 255.0
    # print('归一化的x_train: \n', x_train)
    # print('归一化的x_test: \n', x_test)

    # 将标签转换为独热编码
    y_train = tf.one_hot(y_train, 10)
    y_test = tf.one_hot(y_test, 10)
    # print('独热编码y_train: \n', y_train)
    # print('独热编码y_test: \n', y_test)

    return x_train, y_train, x_test, y_test


def load_data_from_folders(folder_path):
    images = []
    labels = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.jpg'):
                img_path = os.path.join(root, file)
                img = cv2.imread(img_path, 0)
                img = cv2.resize(img, (100, 100))
                img = img.astype('float32') / 255.0
                images.append(img)
                labels.append(os.path.basename(root))
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)
    # print(labels)
    print(len(np.array(labels)))  # 102张照片

    return np.array(images), tf.one_hot(np.array(labels), labels[-1]+1)


if __name__ == '__main__':
    # x_train, y_train, x_test, y_test = load_data()
    # print(type(x_train))  # <class 'numpy.ndarray'>
    # print(type(x_test))  # <class 'numpy.ndarray'>
    # print(type(y_train))  # <class 'tensorflow.python.framework.ops.EagerTensor'>
    # print(type(y_test))  # <class 'tensorflow.python.framework.ops.EagerTensor'>
    # print(len(y_train[1]))
    # print(y_test)
    # print(len(y_test))
    # print(len(y_test[1]))

    x_train, y_train = load_data_from_folders('train')
    # print(y_train[0])
    # np.set_printoptions(threshold=np.inf)
    # print(np.shape(y_train))

    model = creat_model()
    # 编译模型
    model.compile(optimizer="adam", loss="categorical_crossentropy",
                  metrics=["accuracy"])
    # 训练模型                   批次大小=128    迭代次数=25     验证集比例=0.1
    model.fit(x_train, y_train, batch_size=128, epochs=25, validation_split=0.1)

    # print(model.summary())
    y_test=tf.one_hot([1,1,1,1], 7)
    x_test=[]
    img = cv2.imread('test/3.jpg', 0)
    img = cv2.resize(img, (100, 100))
    img = img.astype('float32') / 255.0
    x_test.append(img)
    img = cv2.imread('test/4.jpg', 0)
    img = cv2.resize(img, (100, 100))
    img = img.astype('float32') / 255.0
    x_test.append(img)
    img = cv2.imread('test/5.jpg', 0)
    img = cv2.resize(img, (100, 100))
    img = img.astype('float32') / 255.0
    x_test.append(img)
    img = cv2.imread('test/16.jpg', 0)
    img = cv2.resize(img, (100, 100))
    img = img.astype('float32') / 255.0
    x_test.append(img)
    x_test=np.array(x_test)

    # 在测试集上评估模型
    test_loss, test_accuracy = model.evaluate(x_test, y_test)

    print(f"Test Loss: {test_loss}")
    print(f"Test Accuracy: {test_accuracy}")

    # # for layer in model.layers:
    # #     print(layer.output_shape)
