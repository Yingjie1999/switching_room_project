import cv2
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

def img32_1(file_path):
    img = cv2.imread(file_path)
    h, w = img.shape[:2]
    box = np.array([0.2*w, 0, 0.8*w, h]).astype(np.int) # xmin, ymin, xmax, ymax
    img = img[box[1]:box[3], box[0]:box[2]]
    cv2_show('img', img)
    f_xy = 0.2
    img1 = cv2.resize(img.copy(), None, fx=f_xy, fy=f_xy)
    hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)  # 色彩空间转换为hsv，便于分离
    lower_hsv1 = np.array([50, 43, 46]) # 提取颜色的低值 green [35, 43, 46]
    high_hsv1 = np.array([100, 255, 255]) # 提取颜色的高值 [77, 255, 255]
    mask1 = cv2.inRange(hsv, lowerb=lower_hsv1, upperb=high_hsv1)
    lower_hsv1 = np.array([156, 43, 150]) # 提取颜色的低值 red [156, 43, 46]
    high_hsv1 = np.array([180, 255, 255]) # 提取颜色的高值 [180, 255, 255]
    lower_hsv2 = np.array([0, 43, 150]) # 提取颜色的低值 red [0, 43, 46]
    high_hsv2 = np.array([10, 255, 255]) # 提取颜色的高值 [10, 255, 255]
    mask_1 = cv2.inRange(hsv, lowerb=lower_hsv1, upperb=high_hsv1)
    mask_2 = cv2.inRange(hsv, lowerb=lower_hsv2, upperb=high_hsv2)
    mask2 = cv2.bitwise_or(mask_1, mask_2)
    mask3 = cv2.bitwise_or(mask1, mask2)
    cv2_show('1', img)
    cv2_show('m1', mask3)
    X = []
    for i in range(mask3.shape[0]):
        for j in range(mask3.shape[1]):
            if mask3[i, j] > 10: # y x
                X.append([i, j])
    X = np.array(X)
    db = DBSCAN(eps=4, min_samples=5, metric='euclidean') # 密度聚类DBSCAN 半径，样本点数量，欧式距离
    y_db = db.fit_predict(X)
    num = {}
    for i in range(max(y_db)+1):
        num[i] = len(X[y_db==i, :])
    num = dict(sorted(num.items(), key=lambda item: item[1], reverse=True)) # 对字典的值排序
    X_obj = []
    for key in list(num.keys())[:4]:
        if num[key] > 150:
            X_obj.append(np.average(X[y_db == key, :], axis=0))
        else:
            raise ValueError("没有拍完整")
    X_obj = np.array(X_obj)
    print(X_obj)
    y, x = np.average(X_obj, axis=0)
    h, w = img1.shape[:2]
    box = np.array([[0,0,x,y],[x,0,w,y],[0,y,x,h],[x,y,w,h]]).astype(np.int) # xmin, ymin, xmax, ymax
    # X_order = []
    # for i in range(X_obj.shape[0]):
    #     if X_obj[i, 0] < X_obj_avg[0] and X_obj[i, 1] < X_obj_avg[1]:
    #         X_order.append(0)
    #     elif X_obj[i, 0] < X_obj_avg[0] and X_obj[i, 1] > X_obj_avg[1]:
    #         X_order.append(1)
    #     elif X_obj[i, 0] > X_obj_avg[0] and X_obj[i, 1] < X_obj_avg[1]:
    #         X_order.append(2)
    #     elif X_obj[i, 0] > X_obj_avg[0] and X_obj[i, 1] > X_obj_avg[1]:
    #         X_order.append(3)
    # X_obj = X_obj[X_order]
    results = {}
    for m in range(4):
        img2 = img1[box[m, 1]:box[m, 3], box[m, 0]:box[m, 2]]
        hsv = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)  # 色彩空间转换为hsv，便于分离
        lower_hsv1 = np.array([50, 43, 46])  # 提取颜色的低值 green [35, 43, 46]
        high_hsv1 = np.array([100, 255, 255])  # 提取颜色的高值 [77, 255, 255]
        mask1 = cv2.inRange(hsv, lowerb=lower_hsv1, upperb=high_hsv1)
        X = []
        for i in range(mask1.shape[0]):
            for j in range(mask1.shape[1]):
                if mask1[i, j] > 10:  # y x
                    X.append([i, j])
        X = np.array(X)
        results['{}路调度号'.format(m + 1)] = str(415 + m)
        if len(X)==0:
            results['{}路合分闸指示'.format(m + 1)] = '红色'
            results['{}路开关'.format(m + 1)] = 'i'
            continue
        db = DBSCAN(eps=4, min_samples=5, metric='euclidean')  # 密度聚类DBSCAN 半径，样本点数量，欧式距离
        y_db = db.fit_predict(X)
        num = {}
        for i in range(max(y_db) + 1):
            num[i] = len(X[y_db == i, :])
        num = dict(sorted(num.items(), key=lambda item: item[1], reverse=True))  # 对字典的值排序
        print(num)
        if len(num)==0:
            results['{}路合分闸指示'.format(m + 1)] = '红色'
            results['{}路开关'.format(m + 1)] = 'i'
            continue
        if num[list(num.keys())[0]] > 200:
            results['{}路合分闸指示'.format(m + 1)] = '绿色'
            results['{}路开关'.format(m + 1)] = 'o'
        else:
            results['{}路合分闸指示'.format(m + 1)] = '红色'
            results['{}路开关'.format(m + 1)] = 'i'
        print(num)
    return results

def img32_2(file_path):
    img = cv2.imread(file_path)
    f_xy = 0.2
    img1 = cv2.resize(img.copy(), None, fx=f_xy, fy=f_xy)
    hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)  # 色彩空间转换为hsv，便于分离
    lower_hsv1 = np.array([156, 60, 100]) # 提取颜色的低值 red [156, 43, 46]
    high_hsv1 = np.array([180, 255, 255]) # 提取颜色的高值 [180, 255, 255]
    lower_hsv2 = np.array([0, 60, 100]) # 提取颜色的低值 red [0, 43, 46]
    high_hsv2 = np.array([10, 255, 255]) # 提取颜色的高值 [10, 255, 255]
    mask_1 = cv2.inRange(hsv, lowerb=lower_hsv1, upperb=high_hsv1)
    mask_2 = cv2.inRange(hsv, lowerb=lower_hsv2, upperb=high_hsv2)
    mask1 = cv2.bitwise_or(mask_1, mask_2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))  # 定义结构元素的形状和大小
    mask1 = cv2.dilate(mask1, kernel) # 膨胀操作
    mask1 = cv2.erode(mask1, kernel) # 腐蚀操作
    cv2_show('1', img)
    cv2_show('m1', mask1)
    X = []
    for i in range(mask1.shape[0]):
        for j in range(mask1.shape[1]):
            if mask1[i, j] > 10:
                X.append([i, j])
    X = np.array(X)
    db = DBSCAN(eps=3, min_samples=5, metric='euclidean') # 密度聚类DBSCAN 半径，样本点数量，欧式距离
    y_db = db.fit_predict(X)
    num = {}
    for i in range(max(y_db)+1):
        num[i] = len(X[y_db==i, :])
    print(num)
    obj = None
    for key,value in num.items():
        if value == max(num.values()):
            obj = int(key)
    if obj is not None:
        X_obj = X[y_db == obj, :]
    else:
        raise ValueError
    box = np.array([min(X_obj[:, 1]), min(X_obj[:, 0]), max(X_obj[:, 1]), max(X_obj[:, 0])]) # xmin, ymin, xmax, ymax
    box = (box / f_xy).astype(np.int)
    xy_b = int(0.06*(box[2]-box[0]))
    img_obj = img[box[1]-xy_b:box[3]+xy_b, box[0]-xy_b:box[2]+xy_b]
    print(box)
    cv2_show('3', img_obj)
    # gray_obj = cv2.cvtColor(img_obj, cv2.COLOR_BGR2GRAY)
    # ret1, mask2 = cv2.threshold(gray_obj, 0, 255, cv2.THRESH_OTSU) # 方法选择为THRESH_OTSU
    # cv2_show('000', mask2) # 000
    return  img_obj

def Point_thirty_two(file_name1, file_name2):
    Point_info = []
    results1 = img32_1(file_name1)
    # result_imgs = img32_2(file_name2)
    # cv2_show('result', result_imgs)
    # results1['电流'] = dispatchNum_recog(result_imgs)
    print(results1)
    Point_list = list(results1.values())
    print(Point_list)
    return Point_list
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

if __name__ == '__main__':
    img1_path = './images3/34-1.JPG' # Need to modify!!
    img2_path = './images3/34-2.JPG' # Need to modify!!
    results1 = img32_1(img1_path)
    print(results1)
    # results2 = img34_2(img2_path)
    # cv2_show('result', results2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
