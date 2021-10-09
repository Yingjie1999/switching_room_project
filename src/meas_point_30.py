import cv2
import os
import numpy as np
from sklearn.cluster import DBSCAN
import recognition
#
# def dispatchNum_recog(img):
#     txt = recognition.serial_recogn(img)
#     print(txt)
#     return txt

def cv2_show(name, img_):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, img_)

def img30_1(file_path):
    img = cv2.imread(file_path)
    f_xy = 0.1
    img1 = cv2.resize(img.copy(), None, fx=f_xy, fy=f_xy)

    hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)  # 色彩空间转换为hsv，便于分离
    lower_hsv1 = np.array([100, 43, 100])  # 提取颜色的低值 blue [100, 43, 46]
    high_hsv1 = np.array([124, 255, 255])  # 提取颜色的高值 [124, 255, 255]
    mask1 = cv2.inRange(hsv, lowerb=lower_hsv1, upperb=high_hsv1)
    X = []
    for i in range(mask1.shape[0]):
        for j in range(mask1.shape[1]):
            if mask1[i, j] > 10: # y x
                X.append([i, j])
    X = np.array(X)
    db = DBSCAN(eps=4, min_samples=5, metric='euclidean') # 密度聚类DBSCAN 半径，样本点数量，欧式距离
    y_db = db.fit_predict(X)
    num = {}
    for i in range(max(y_db)+1):
        num[i] = len(X[y_db==i, :])
    num = dict(sorted(num.items(), key=lambda item: item[1], reverse=True)) # 对字典的值排序
    print(num)
    X_obj = X[y_db == list(num.keys())[0], :]
    box = np.array([min(X_obj[:, 1]), min(X_obj[:, 0]), max(X_obj[:, 1]), max(X_obj[:, 0])]) # xmin, ymin, xmax, ymax
    box = (box / f_xy).astype(np.int)
    img_obj1 = img[box[1]:box[3], box[0]:box[2]]
    r = box[2]-box[0]
    box = np.array([box[0]-0.5*r, box[1]-1.5*r, box[0]+1.4*r, box[1]-0.2*r]).astype(np.int)
    print(box)
    img_obj2 = img[box[1]:box[3], box[0]:box[2]]
    #cv2_show#('1', mask1)
    #cv2_show#('2', img_obj1)
    #cv2_show#('3', img_obj2)

    img_objs = [img_obj2]  # ------------
    hsv = cv2.cvtColor(img_obj1, cv2.COLOR_BGR2HSV)  # 色彩空间转换为hsv，便于分离
    lower_hsv1 = np.array([10, 43, 100])  # 提取颜色的低值 yellow [26, 43, 46]
    high_hsv1 = np.array([34, 255, 255])  # 提取颜色的高值 [34, 255, 255]
    mask1 = cv2.inRange(hsv, lowerb=lower_hsv1, upperb=high_hsv1)

    h, w = mask1.shape[:2]
    obj = mask1[int(0.2*h):int(0.8*h), int(0*w):int(1*w)]
    img_objs.append(obj)
    #cv2_show#('4', obj)
    return img_objs

def tran(list):
    list_str = str(list).replace("[", "").replace("]", "")
    return eval(f"[{list_str}]")

def Point_thirty(file_name1):
    # Point_info = []
    # results = {}
    # result_imgs = img30_1(file_name1)
    # for i in range(len(result_imgs)):
    #     #cv2_show#('obj{}'.format(i), result_imgs[i])
    # # results['变压器温度A'] = recognition.digit_detect(result_imgs[1])
    # # results['调度号'] = dispatchNum_recog(result_imgs[0])
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    Point_list = []
    Point_list.append([' '])
    print("Point_thirty", Point_list)
    return tran(Point_list)
#
# if __name__ == '__main__':
#     img1_path = './images2/30-1.JPG'
#     result_imgs = img30_1(img1_path)
#     for i in range(len(result_imgs)):
#         #cv2_show#('obj{}'.format(i), result_imgs[i])
#     # results['变压器温度A'] = dispatchNum_recog(result_imgs[1])
#     # results['调度号'] = dispatchNum_recog(result_imgs[0])
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
