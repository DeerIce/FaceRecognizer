# FaceRecognizer
A model that can recognize faces and genders.



优化：对于训练集路径D:/Code/FaceRecognizer/train/Armas，也许里面有很多其它的文件，而我们需要的只是图片文件（如jpg文件），所以要在该路径中寻找file.endswith('.jpg')，即以jpg为扩展名的文件。



GUI2.py中关于filepath = os.path.join(root, filename)，把照片名（即01.jpg、02.jpg……）拼接到根路径root时，可能会出现路径斜杠不一致的现象，如D:/Code/FaceRecognizer/train/Armas\3.jpg



root包含了所选文件里所有文件夹的路径，并且子文件夹都是以\分隔，之前的部分就是/分隔，如

D:/Code/FaceRecognizer   \train\Gloria
