# 1. 导入必要的库
import cv2
import numpy as np
from scipy.io import loadmat
from sklearn.svm import SVC

# 2. 加载数据并做处理
# 加载图片的数据并转换为灰度图
img_1 = cv2.imread('train/Gloria/00001.jpg', 0)
img_2 = cv2.imread('train/Armas/1.jpg', 0)
img_3 = cv2.imread('train/Gloria/00002.jpg', 0)

# 使用滤波技术对图片和脸部信息做出优化
# 设置高斯参数
gaussian_ksize = (3, 3)
gaussian_sigma = 0.1
# 使用高斯滤波器对图片和脸部进行滤波
img1_gaussian = cv2.GaussianBlur(img_1, gaussian_ksize, gaussian_sigma)
img2_gaussian = cv2.GaussianBlur(img_2, gaussian_ksize, gaussian_sigma)
img3_gaussian = cv2.GaussianBlur(img_3, gaussian_ksize, gaussian_sigma)

# 使用PCA(主成分分析）技术和 SVM（支持向量机）对图片和脸部信息做处理
# 获取脸部提取特征向量
eigenfaces = loadmat('eigenfaces.mat')
# 使用PCA方法提取人脸的特征向量
faces1_pca, faces2_pca = [
    np.dot(img1_gaussian, face.transpose())/255 for face in eigenfaces]
# 开始训练支持向量机
svm_classifier = SVC(gamma='auto').fit(faces1_pca, faces2_pca)

# 3. 运行机器学习模型
# 使用训练好的模型来分类图片中的人脸
# 获取特征向量
faces3_pca = np.dot(img3_gaussian, eigenfaces.transpose())/255
# 对输入图片进行分类
predicted_class = svm_classifier.predict(faces3_pca)

# 4. 输出结果
# 输出分类结果
print("The predicted class is: {0}".format(predicted_class))
