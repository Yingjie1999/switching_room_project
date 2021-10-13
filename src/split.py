import cv2
import os
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

# def cv2.imshow(name, img_):
#     cv2.namedWindow(name)
#     #cv2.imshow#(name, img_)

def detect_and_split_1(image): # 红色数码管
    img = image
    f_xy = 0.2
    img1 = cv2.resize(img.copy(), None, fx=f_xy, fy=f_xy)
    hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)  # 色彩空间转换为hsv，便于分离
    lower_hsv1 = np.array([156, 100, 100])  # 提取颜色的低值 red [156, 43, 46]
    high_hsv1 = np.array([180, 255, 255])  # 提取颜色的高值 [180, 255, 255]
    lower_hsv2 = np.array([0, 100, 100])  # 提取颜色的低值 red [0, 43, 46]
    high_hsv2 = np.array([10, 255, 255])  # 提取颜色的高值 [10, 255, 255]
    mask_1 = cv2.inRange(hsv, lowerb=lower_hsv1, upperb=high_hsv1)
    mask_2 = cv2.inRange(hsv, lowerb=lower_hsv2, upperb=high_hsv2)
    mask1 = cv2.bitwise_or(mask_1, mask_2)
    lower_hsv2 = np.array([0, 0, 0])  # 提取颜色的低值 black [0, 0, 0]
    high_hsv2 = np.array([180, 255, 80])  # 提取颜色的高值 [180, 255, 46]
    mask2 = cv2.inRange(hsv, lowerb=lower_hsv2, upperb=high_hsv2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 30))  # 定义结构元素的形状和大小
    dst = cv2.dilate(mask1, kernel)  # 膨胀操作
    mask_and = cv2.bitwise_and(dst, mask2)
    cv2.imshow('1', mask1)
    cv2.imshow('2', mask2)
    X = []
    for i in range(mask_and.shape[0]):
        for j in range(mask_and.shape[1]):
            if mask_and[i, j] > 10:
                X.append([i, j])
    X = np.array(X)
    print(np.shape(X))
    db = DBSCAN(eps=3, min_samples=5, metric='euclidean')  # 密度聚类DBSCAN 半径，样本点数量，欧式距离
    y_db = db.fit_predict(X)
    num = {}
    for i in range(max(y_db) + 1):
        num[i] = len(X[y_db == i, :])
    obj = None
    for key, value in num.items():
        if value == max(num.values()):
            obj = int(key)
    if obj is not None:
        X_obj = X[y_db == obj, :]
    else:
        raise ValueError
    # print(X_obj)
    print(np.shape(X_obj))
    box = np.array([min(X_obj[:, 1]), min(X_obj[:, 0]), max(X_obj[:, 1]), max(X_obj[:, 0])])  # xmin, ymin, xmax, ymax
    box = (box / f_xy).astype(np.int)
    img_obj = img[box[1]:box[3], box[0]:box[2]]
    cv2.imshow('3', img_obj)

    hsv_obj = cv2.cvtColor(img_obj, cv2.COLOR_BGR2HSV)  # 色彩空间转换为hsv，便于分离
    lower_hsv1 = np.array([156, 100, 100])  # 提取颜色的低值 red [156, 43, 46]
    high_hsv1 = np.array([180, 255, 255])  # 提取颜色的高值 [180, 255, 255]
    lower_hsv2 = np.array([0, 100, 100])  # 提取颜色的低值 red [0, 43, 46]
    high_hsv2 = np.array([10, 255, 255])  # 提取颜色的高值 [10, 255, 255]
    mask_1 = cv2.inRange(hsv_obj, lowerb=lower_hsv1, upperb=high_hsv1)
    mask_2 = cv2.inRange(hsv_obj, lowerb=lower_hsv2, upperb=high_hsv2)
    mask3 = cv2.bitwise_or(mask_1, mask_2)
    cv2.imshow('4', mask3)

    X2 = []
    for i in range(mask3.shape[0]):
        for j in range(mask3.shape[1]):
            if mask3[i, j] > 10:
                X2.append([i, j])
    X2 = np.array(X2)
    box2 = np.array([min(X2[:, 1]), min(X2[:, 0]), max(X2[:, 1]), max(X2[:, 0])])  # xmin, ymin, xmax, ymax
    img_obj2 = mask3[box2[1]:box2[3], box2[0]:box2[2]]
    cv2.imshow('5', img_obj2)

    h, w = img_obj2.shape[:2]
    x_b, y_b, y_h = 0.1 * w, 0.1 * h, (0.2 * h + h) / 3
    box_img = []
    for i in range(3):
        box_img.append([box2[0] - x_b, box2[1] - y_b + i * y_h, box2[2] + x_b, box2[1] - y_b + (i + 1) * y_h])
    box_img = np.array(box_img).astype(np.int)

    img_objs = []
    for i in range(3):
        char = img_obj[box_img[i, 1]:box_img[i, 3], box_img[i, 0]:box_img[i, 2]]
        img_objs.append(char)
    # for i in range(len(img_objs)):
        cv2.imshow('obj{}'.format(i), img_objs[i])
    return img_objs

def detect_and_split_2(file_path): # 绿色液晶屏
    img = cv2.imread(file_path)
    f_xy = 0.2
    img1 = cv2.resize(img.copy(), None, fx=f_xy, fy=f_xy)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)  # 色彩空间转换为hsv，便于分离
    lower_hsv1 = np.array([35, 43, 46])  # 提取颜色的低值 green [35, 43, 46]
    high_hsv1 = np.array([77, 255, 255])  # 提取颜色的高值 [77, 255, 255]
    mask1 = cv2.inRange(hsv, lowerb=lower_hsv1, upperb=high_hsv1)
    lower_hsv2 = np.array([0, 0, 0])  # 提取颜色的低值 black [0, 0, 0]
    high_hsv2 = np.array([180, 255, 80])  # 提取颜色的高值 [180, 255, 46]
    mask2 = cv2.inRange(hsv, lowerb=lower_hsv2, upperb=high_hsv2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 30))  # 定义结构元素的形状和大小
    dst = cv2.dilate(mask2, kernel)  # 膨胀操作
    mask_and = cv2.bitwise_and(dst, mask1)
    print(mask_and.shape)
    X = []
    for i in range(mask_and.shape[0]):
        for j in range(mask_and.shape[1]):
            if mask_and[i, j] > 10:
                X.append([i, j])
    X = np.array(X)
    db = DBSCAN(eps=3, min_samples=5, metric='euclidean')  # 密度聚类DBSCAN 半径，样本点数量，欧式距离
    y_db = db.fit_predict(X)
    num = {}
    for i in range(max(y_db) + 1):
        num[i] = len(X[y_db == i, :])
    print(num)
    obj = None
    for key, value in num.items():
        if value == max(num.values()):
            obj = int(key)
    if obj is not None:
        X_obj = X[y_db == obj, :]
    else:
        raise ValueError
    box = np.array([min(X_obj[:, 1]), min(X_obj[:, 0]), max(X_obj[:, 1]), max(X_obj[:, 0])])  # xmin, ymin, xmax, ymax
    box_r = (box / f_xy).astype(np.int)
    img_obj = img[box_r[1]:box_r[3], box_r[0]:box_r[2]]
    # cv2.imshow('1', mask1)
    # cv2.imshow('2', mask2)
    cv2.imshow('3', img_obj)
    # cv2.imwrite('333.jpg',img_obj)
    # hsv_obj = cv2.cvtColor(img_obj, cv2.COLOR_BGR2HSV)  # 色彩空间转换为hsv，便于分离
    # lower_hsv2 = np.array([0, 0, 0])  # 提取颜色的低值 black [0, 0, 0]
    # high_hsv2 = np.array([180, 255, 100])  # 提取颜色的高值 [180, 255, 46]
    # mask3 = cv2.inRange(hsv_obj, lowerb=lower_hsv2, upperb=high_hsv2)
    gray_obj = cv2.cvtColor(img_obj, cv2.COLOR_BGR2GRAY)
    ret1, mask3 = cv2.threshold(gray_obj, 0, 255, cv2.THRESH_OTSU)  # 方法选择为THRESH_OTSU
    # cv2.imshow('4', mask3)

    X2 = []
    X3 = {}
    for i in range(mask3.shape[0]):
        X3[i] = 0
        for j in range(mask3.shape[1]):
            if mask3[i, j] > 10:
                X2.append([i, j])
                X3[i] = X3[i] + 1
    X2 = np.array(X2)
    box2 = np.array([min(X2[:, 1]), min(X2[:, 0]), max(X2[:, 1]), max(X2[:, 0])])  # xmin, ymin, xmax, ymax
    img_obj2 = mask3[box2[1]:box2[3], box2[0]:box2[2]]
    # cv2.imshow('5', img_obj2)
    print(X3)

    h = img_obj2.shape[0] // 4
    img_objs = []
    for i in range(4):
        xmin = int(0 + box2[0] + box_r[0])
        ymin = int(i * h + box2[1] + box_r[1] - 0.1 * h)
        xmax = int(img_obj2.shape[1] + box2[0] + box_r[0])
        ymax = int((i + 1) * h + box2[1] + box_r[1] + 0.1 * h)
        char = img[ymin:ymax, xmin:xmax]
        img_objs.append(char)

    img_obj3 = []
    for i in range(4):
        quarter = img_obj2[i * h:(i + 1)*h,10:img_obj2.shape[1]]#这里要注意一下，后面需要把边框给去除
        img_obj3.append(quarter)

    # for i in range(len(img_objs)):
        # cv2.imshow('obj{}'.format(i), img_objs[i])
        # cv2.imshow('obj_er{}'.format(i),img_obj3[i])
        # cv2.imwrite('obj{}'.format(i)+'.jpg', img_objs[i])
    return img_obj

def detect_and_split_3(file_path): # 白色字
    img = cv2.imread(file_path)
    f_xy = 0.2
    img1 = cv2.resize(img.copy(), None, fx=f_xy, fy=f_xy)

    hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)  # 色彩空间转换为hsv，便于分离
    lower_hsv1 = np.array([0, 0, 100])  # 提取颜色的低值 white [0, 0, 221]
    high_hsv1 = np.array([180, 5, 255])  # 提取颜色的高值 [180, 30, 255]
    mask1 = cv2.inRange(hsv, lowerb=lower_hsv1, upperb=high_hsv1)
    lower_hsv2 = np.array([100, 43, 0])  # 提取颜色的低值 blue [100, 43, 46] blue & black
    high_hsv2 = np.array([124, 255, 255])  # 提取颜色的高值 [124, 255, 255]
    mask2 = cv2.inRange(hsv, lowerb=lower_hsv2, upperb=high_hsv2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 30))  # 定义结构元素的形状和大小
    dst = cv2.dilate(mask1, kernel)  # 膨胀操作
    mask_and = cv2.bitwise_and(dst, mask2)
    print(mask_and.shape)
    X = []
    for i in range(mask_and.shape[0]):
        for j in range(mask_and.shape[1]):
            if mask_and[i, j] > 10:
                X.append([i, j])
    X = np.array(X)
    db = DBSCAN(eps=3, min_samples=5, metric='euclidean')  # 密度聚类DBSCAN 半径，样本点数量，欧式距离
    y_db = db.fit_predict(X)
    num = {}
    for i in range(max(y_db) + 1):
        num[i] = len(X[y_db == i, :])
    print(num)
    obj = None
    for key, value in num.items():
        if value == max(num.values()):
            obj = int(key)
    if obj is not None:
        X_obj = X[y_db == obj, :]
    else:
        raise ValueError
    box = np.array([min(X_obj[:, 1]), min(X_obj[:, 0]), max(X_obj[:, 1]), max(X_obj[:, 0])])  # xmin, ymin, xmax, ymax
    box_r = (box / f_xy).astype(np.int)
    img_obj = img[box_r[1]:box_r[3], box_r[0]:box_r[2]]
    # cv2.imshow('1', mask1)
    # cv2.imshow('2', mask2)
    # cv2.imshow('3', img_obj)

    # hsv_obj = cv2.cvtColor(img_obj, cv2.COLOR_BGR2HSV)  # 色彩空间转换为hsv，便于分离
    # mask3 = cv2.inRange(hsv_obj, lowerb=lower_hsv1, upperb=high_hsv1)
    gray_obj = cv2.cvtColor(img_obj, cv2.COLOR_BGR2GRAY)
    ret1, mask3 = cv2.threshold(gray_obj, 0, 255, cv2.THRESH_OTSU)  # 方法选择为THRESH_OTSU
    mask_obj = mask3[mask3.shape[0]//8:mask3.shape[0]//8*7,0:mask3.shape[1]]
    # cv2.imshow('4', mask_obj)
    # cv2.imwrite('AH2.jpg',mask_obj)

    X3 = {}
    for i in range(mask3.shape[1]):
        X3[i] = 0
        for j in range(mask3.shape[0]):
            if mask3[j, i] > 10:
                X3[i] = X3[i] + 1
    del_key1 = [0]
    for key, value in X3.items():
        if value < 10:
            del_key1.append(key)
        else:
            break
    X3_keys = sorted(X3, reverse=True)
    del_key2 = [mask3.shape[1]]
    for key in X3_keys:
        if X3[key] < 10:
            del_key2.append(key)
        else:
            break
    pos = [max(del_key1), min(del_key2)]
    print(pos)
    img_obj2 = img_obj[1:img_obj.shape[0] - 1, pos[0] - 30:pos[1] + 30]  # xmin, ymin, xmax, ymax
    cv2.imshow('5', img_obj2)
    # cv2.imwrite('AH222.jpg', img_obj2)
    # return img_obj2
    return img_obj2




if __name__ == '__main__':
    img1_path = './images/4.jpg'
    detect_and_split_1(img1_path)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # img2_path = './images/2_1.jpg'
    # image = detect_and_split_2(img2_path)
    # # for i in range(len(image)):
    # #     cv2.imshow('obj_lcd{}'.format(i), image[i])
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # img3_path = './images/5.jpg'
    # detect_and_split_3(img3_path)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()