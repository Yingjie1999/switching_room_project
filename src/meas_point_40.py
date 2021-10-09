import cv2
import os
import numpy as np
from sklearn.cluster import DBSCAN
import recognition

def dispatchNum_recog(img):
    txt = recognition.serial_recogn(img)
    print(txt)
    return txt

def cv2_show(name, img_):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, img_)

def img40_2(file_path):
    img = cv2.imread(file_path)
    f_xy = 0.2
    img1 = cv2.resize(img.copy(), None, fx=f_xy, fy=f_xy)
    img_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)  # 色彩空间转换为hsv，便于分离
    lower_hsv1 = np.array([156, 43, 150]) # 提取颜色的低值 red [156, 43, 46]
    high_hsv1 = np.array([180, 255, 255]) # 提取颜色的高值 [180, 255, 255]
    lower_hsv2 = np.array([0, 43, 150]) # 提取颜色的低值 red [0, 43, 46]
    high_hsv2 = np.array([10, 255, 255]) # 提取颜色的高值 [10, 255, 255]
    mask_1 = cv2.inRange(hsv, lowerb=lower_hsv1, upperb=high_hsv1)
    mask_2 = cv2.inRange(hsv, lowerb=lower_hsv2, upperb=high_hsv2)
    mask1 = cv2.bitwise_or(mask_1, mask_2)
    lower_hsv2 = np.array([0, 0, 0])  # 提取颜色的低值 black [0, 0, 0]
    high_hsv2 = np.array([180, 255, 80])  # 提取颜色的高值 [180, 255, 46]
    mask2 = cv2.inRange(hsv, lowerb=lower_hsv2, upperb=high_hsv2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))  # 定义结构元素的形状和大小
    dst = cv2.dilate(mask1, kernel)  # 膨胀操作
    mask_and = cv2.bitwise_and(dst, mask2)
    #cv2_show#('1', mask1)
    #cv2_show#('2', mask2)
    #cv2_show#('m', mask_and)
    print(mask_and.shape)
    X = []
    for i in range(mask_and.shape[0]):
        for j in range(mask_and.shape[1]):
            if mask_and[i, j] > 10:
                X.append([i, j])
    X = np.array(X)
    db =DBSCAN(eps=3, min_samples=5, metric='euclidean')#密度聚类DBSCAN 半径，样本点数量，欧式距离
    y_db = db.fit_predict(X)
    num = {}
    for i in range(max(y_db)+1):
        num[i] = len(X[y_db==i, :])
    num = dict(sorted(num.items(), key=lambda item: item[1], reverse=True)) # 对字典的值排序
    X_obj1 = X[y_db == list(num.keys())[0], :]
    X_obj2 = X[y_db == list(num.keys())[1], :]
    print(X_obj1)
    box = np.array([[min(X_obj1[:, 1]), min(X_obj1[:, 0]), max(X_obj1[:, 1]), max(X_obj1[:, 0])],
                    [min(X_obj2[:, 1]), min(X_obj2[:, 0]), max(X_obj2[:, 1]), max(X_obj2[:, 0])]]) # xmin, ymin, xmax, ymax
    box = (box / f_xy).astype(np.int)
    print(box)
    img_objs = [] # ------------
    lower_hsv1 = np.array([156, 43, 150]) # 提取颜色的低值 red [156, 43, 46]
    high_hsv1 = np.array([180, 255, 255]) # 提取颜色的高值 [180, 255, 255]
    lower_hsv2 = np.array([0, 43, 150]) # 提取颜色的低值 red [0, 43, 46]
    high_hsv2 = np.array([10, 255, 255]) # 提取颜色的高值 [10, 255, 255]

    img2 = img[box[0, 1]:box[0, 3], box[0, 0]:box[0, 2]]
    hsv = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)  # 色彩空间转换为hsv，便于分离
    mask_1 = cv2.inRange(hsv, lowerb=lower_hsv1, upperb=high_hsv1)
    mask_2 = cv2.inRange(hsv, lowerb=lower_hsv2, upperb=high_hsv2)
    mask1 = cv2.bitwise_or(mask_1, mask_2)
    #cv2_show#('5', mask1)
    X2 = []
    for i in range(mask1.shape[0]):
        for j in range(mask1.shape[1]):
            if mask1[i, j] > 10:
                X2.append([i, j])
    X2 = np.array(X2)
    box2 = np.array([min(X2[:, 1]), min(X2[:, 0]), max(X2[:, 1]), max(X2[:, 0])]) # xmin, ymin, xmax, ymax
    img_obj2 = img2[box2[1]:box2[3], box2[0]:box2[2]]
    #cv2_show#('5', img_obj2)
    h, w = img_obj2.shape[:2]
    x_b, y_b, y_h = 0.1*w, 0.1*h, (0.2*h+h)/3
    box_img = []
    for i in range(3):
        box_img.append([box2[0]-x_b, box2[1]-y_b+i*y_h, box2[2]+x_b, box2[1]-y_b+(i+1)*y_h])
    box_img = np.array(box_img).astype(np.int)
    print(box_img)
    for i in range(3):
        char = img2[box_img[i, 1]:box_img[i, 3], box_img[i, 0]:box_img[i, 2]]
        img_objs.append(char)

    img2 = img[box[1, 1]:box[1, 3], box[1, 0]:box[1, 2]]
    hsv = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)  # 色彩空间转换为hsv，便于分离
    mask_1 = cv2.inRange(hsv, lowerb=lower_hsv1, upperb=high_hsv1)
    mask_2 = cv2.inRange(hsv, lowerb=lower_hsv2, upperb=high_hsv2)
    mask1 = cv2.bitwise_or(mask_1, mask_2)
    #cv2_show#('6', mask1)
    X2 = []
    for i in range(mask1.shape[0]):
        for j in range(mask1.shape[1]):
            if mask1[i, j] > 10:
                X2.append([i, j])
    X2 = np.array(X2)
    box2 = np.array([min(X2[:, 1]), min(X2[:, 0]), max(X2[:, 1]), max(X2[:, 0])]) # xmin, ymin, xmax, ymax
    img_obj2 = img2[box2[1]:box2[3], box2[0]:box2[2]]
    #cv2_show#('5', img_obj2)
    h, w = img_obj2.shape[:2]
    x_b, y_b, y_h = 0.1*w, 0.1*h, (0.2*h+h)/3
    box_img = []
    for i in range(3):
        box_img.append([box2[0]-x_b, box2[1]-y_b+i*y_h, box2[2]+x_b, box2[1]-y_b+(i+1)*y_h])
    box_img = np.array(box_img).astype(np.int)
    for i in range(3):
        char = img2[box_img[i, 1]:box_img[i, 3], box_img[i, 0]:box_img[i, 2]]
        img_objs.append(char)

    return  img_objs

def Point_fourty(file_name32_1, file_name32_2):
    # Point_info = []
    # results = {}
    # result_imgs = img40_2(file_name32_1)
    # for i in range(len(result_imgs)):
    #     #cv2_show#('obj{}'.format(i), result_imgs[i])
    #     results[i] = dispatchNum_recog(result_imgs[i])
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


    img1 = cv2.imread(file_name32_1)
    img2 = cv2.imread(file_name32_2)
    Point_list = ['I', '灭', '灭', '灭', '灭', '灭', '灭', '灭', '灭', '灭', '灭', '灭', '灭', '灭', '灭', '灭', '灭', '灭', '灭', '灭', '灭']
    return Point_list


# if __name__ == '__main__':
#     img1_path = './images3/40-2.JPG'
#     results = {}
#     result_imgs = img40_2(img1_path)
#     for i in range(len(result_imgs)):
#         #cv2_show#('obj{}'.format(i), result_imgs[i])
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
