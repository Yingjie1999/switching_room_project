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

def img34_1(file_path):
    img = cv2.imread(file_path)
    cv2_show('1', img)
    f_xy = 0.1
    img1 = cv2.resize(img.copy(), None, fx=f_xy, fy=f_xy)
    hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)  # 色彩空间转换为hsv，便于分离
    lower_hsv1 = np.array([0, 0, 0])  # 提取颜色的低值 black [0, 0, 0]
    high_hsv1 = np.array([180, 255, 80])  # 提取颜色的高值 [180, 255, 46]
    mask1 = cv2.inRange(hsv, lowerb=lower_hsv1, upperb=high_hsv1)

    cv2_show('m1', mask1)
    X = []
    for i in range(mask1.shape[0]):
        for j in range(mask1.shape[1]):
            if mask1[i, j] > 10:
                X.append([i, j])
    X = np.array(X)
    db =DBSCAN(eps=2, min_samples=5, metric='euclidean')#密度聚类DBSCAN 半径，样本点数量，欧式距离
    y_db = db.fit_predict(X)
    num = {}
    for i in range(max(y_db)+1):
        num[i] = len(X[y_db==i, :])
    num = dict(sorted(num.items(), key=lambda item: item[1], reverse=True)) # 对字典的值排序
    print(num)
    X_obj = X[y_db == list(num.keys())[0], :]
    x, y = [int((min(X_obj[:, 1])+max(X_obj[:, 1]))/2), min(X_obj[:, 0])]
    r = max(X_obj[:, 1]) - min(X_obj[:, 1])
    box = np.array([x-0.15*r, y-0.2*r, x+0.15*r, y+0.02*r]).astype(np.int) # xmin, ymin, xmax, ymax
    box = (box / f_xy).astype(np.int)
    print(box)
    region = img[box[1]:box[3], box[0]:box[2]]
    cv2_show('o', region)

    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)  # 色彩空间转换为hsv，便于分离
    lower_hsv1 = np.array([156, 43, 46]) # 提取颜色的低值 red [156, 43, 46]
    high_hsv1 = np.array([180, 255, 255]) # 提取颜色的高值 [180, 255, 255]
    lower_hsv2 = np.array([0, 43, 46]) # 提取颜色的低值 red [0, 43, 46]
    high_hsv2 = np.array([10, 255, 255]) # 提取颜色的高值 [10, 255, 255]
    mask_1 = cv2.inRange(hsv, lowerb=lower_hsv1, upperb=high_hsv1)
    mask_2 = cv2.inRange(hsv, lowerb=lower_hsv2, upperb=high_hsv2)
    mask1 = cv2.bitwise_or(mask_1, mask_2)
    cv2_show('m', mask1)
    X = []
    for i in range(mask1.shape[0]):
        for j in range(mask1.shape[1]):
            if mask1[i, j] > 10:
                X.append([i, j])
    X = np.array(X)
    box = np.array([min(X[:, 1]), min(X[:, 0]), max(X[:, 1]), max(X[:, 0])]) # xmin, ymin, xmax, ymax
    xy_b = int(0.3*(box[2]-box[0]))
    obj = region[box[1]-xy_b:box[3]+xy_b, box[0]-xy_b:box[2]+xy_b]
    cv2_show('m2', obj)
    return obj

def img34_2(file_path):
    print("11111")
    results = {}
    # 找到分闸按钮
    img = cv2.imread(file_path)
    h, w = img.shape[:2]
    box = np.array([0.25*w, 0.1*h, 0.8*w, 0.75*h]).astype(np.int) # xmin, ymin, xmax, ymax
    img = img[box[1]:box[3], box[0]:box[2]]
    cv2_show('img', img)
    f_xy = 0.4
    img1 = cv2.resize(img.copy(), None, fx=f_xy, fy=f_xy)
    hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)  # 色彩空间转换为hsv，便于分离
    lower_hsv1 = np.array([156, 43, 100]) # 提取颜色的低值 red [156, 43, 46]
    high_hsv1 = np.array([180, 255, 255]) # 提取颜色的高值 [180, 255, 255]
    lower_hsv2 = np.array([0, 43, 100]) # 提取颜色的低值 red [0, 43, 46]
    high_hsv2 = np.array([10, 255, 255]) # 提取颜色的高值 [10, 255, 255]
    mask_1 = cv2.inRange(hsv, lowerb=lower_hsv1, upperb=high_hsv1)
    mask_2 = cv2.inRange(hsv, lowerb=lower_hsv2, upperb=high_hsv2)
    mask1 = cv2.bitwise_or(mask_1, mask_2)
    mask=cv2.Canny(mask1,30,100)
    cv2_show('m0', mask1)
    cv2_show('m1', mask)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    circles = cv2.HoughCircles(mask,cv2.HOUGH_GRADIENT,1,20,
                                param1=100,param2=20,minRadius=0,maxRadius=100)[0] # 霍夫变换圆检测
    print(circles) # 输出返回值，方便查看类型
    # if circles is None:
    #     raise ValueError('No circles!')
    # else:
    circles = circles[np.argsort(circles[:, 0]), :]
    print(circles[-1, :])
    x, y, r = circles[-1, :]
    results['分闸按钮'] = '灭'

    # 合闸按钮
    box1 = np.array([x-18*r, y-3*r, x-10*r, y+3*r]).astype(np.int)
    obj = img1[box1[1]:box1[3], box1[0]:box1[2]]
    hsv = cv2.cvtColor(obj, cv2.COLOR_BGR2HSV)  # 色彩空间转换为hsv，便于分离
    lower_hsv1 = np.array([35, 100, 46])  # 提取颜色的低值 green [35, 43, 46]
    high_hsv1 = np.array([100, 255, 120])  # 提取颜色的高值 [77, 255, 255]
    mask1 = cv2.inRange(hsv, lowerb=lower_hsv1, upperb=high_hsv1)
    cv2_show('y2', obj)
    cv2_show('y', mask1)
    mask = cv2.Canny(mask1,30,100)
    cv2_show('m2', mask)
    circles = cv2.HoughCircles(mask,cv2.HOUGH_GRADIENT,1,20,
                                param1=100,param2=20,minRadius=0,maxRadius=100) # 霍夫变换圆检测
    print(circles) #输出返回值，方便查看类型
    if circles is None: results['合闸按钮'] = '亮'
    else:
        ind = np.unravel_index(np.argmax(circles[0], axis=None), circles[0].shape)
        r2 = circles[0, ind[0], 2].astype(np.int)
        if r/2 < r2 < r*1.5:
            x2, y2 = circles[0, ind[0], :2].astype(np.int)
            mean = np.sum(obj[y2-r2:y2+r2, x2-r2:x2+r2]) / pow(r2*2,2)
            print(mean)
            if mean < 60: results['合闸按钮'] = '亮'
            else: results['合闸按钮'] = '灭'
        else: results['合闸按钮'] = '亮'

    # 确定合闸压板位置, 检测情况
    print(r)
    box1 = np.array([x-11*r, y-4.3*r, x-3*r, y+4.3*r]).astype(np.int)
    print(box1)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    obj = img1[box1[1]:box1[3], box1[0]:box1[2]]
    cv2_show('y2', obj)

    hsv = cv2.cvtColor(obj, cv2.COLOR_BGR2HSV)  # 色彩空间转换为hsv，便于分离
    lower_hsv1 = np.array([11, 100, 100])  # 提取颜色的低值 yellow [26, 43, 46]
    high_hsv1 = np.array([34, 255, 255])  # 提取颜色的高值 [34, 255, 255]
    mask1 = cv2.inRange(hsv, lowerb=lower_hsv1, upperb=high_hsv1)

    cv2_show('y', mask1)
    X = []
    for i in range(mask1.shape[0]):
        for j in range(mask1.shape[1]):
            if mask1[i, j] > 10:
                X.append([i, j])
    X = np.array(X)
    db = DBSCAN(eps=2, min_samples=5, metric='euclidean') # 密度聚类DBSCAN 半径，样本点数量，欧式距离
    y_db = db.fit_predict(X)
    num = {}
    for i in range(max(y_db)+1):
        num[i] = len(X[y_db==i, :])
    print(num)
    o = None
    for key,value in num.items():
        if value == max(num.values()):
            o = int(key)
    if o is not None:
        X_obj = X[y_db == o, :]
    else:
        raise ValueError
    box = np.array([min(X_obj[:, 1]), min(X_obj[:, 0]), max(X_obj[:, 1]), max(X_obj[:, 0])]) # xmin, ymin, xmax, ymax
    if (box[3]-box[1])/(box[2]-box[0]) > 2:
        results['合闸压板'] = '合'
    else: results['合闸压板'] = '开'

    # 合闸指示
    box1 = np.array([x - 18 * r, y - 10 * r, x - 10 * r, y - 4 * r]).astype(np.int)
    obj = img1[box1[1]:box1[3], box1[0]:box1[2]]
    cv2_show('3', obj)
    hsv = cv2.cvtColor(obj, cv2.COLOR_BGR2HSV) # 色彩空间转换为hsv，便于分离
    lower_hsv1 = np.array([156, 100, 0]) # 提取颜色的低值 red [156, 43, 46]
    high_hsv1 = np.array([180, 255, 120]) # 提取颜色的高值 [180, 255, 255]
    lower_hsv2 = np.array([0, 100, 0]) # 提取颜色的低值 red [0, 43, 46]
    high_hsv2 = np.array([10, 255, 120]) # 提取颜色的高值 [10, 255, 255]
    mask1 = cv2.inRange(hsv, lowerb=lower_hsv1, upperb=high_hsv1)
    mask2 = cv2.inRange(hsv, lowerb=lower_hsv2, upperb=high_hsv2)
    mask = cv2.bitwise_or(mask1, mask2)
    mask=cv2.Canny(mask,30,100)
    cv2_show('m2', mask)
    circles = cv2.HoughCircles(mask,cv2.HOUGH_GRADIENT,1,20,
                                param1=100,param2=20,minRadius=0,maxRadius=100) # 霍夫变换圆检测
    print(circles) #输出返回值，方便查看类型
    if circles is None: results['合闸指示'] = '亮'
    else:
        ind = np.unravel_index(np.argmax(circles[0], axis=None), circles[0].shape)
        r2 = circles[0, ind[0], 2].astype(np.int)
        cv2_show('3', obj)
        if r/2 < r2 < r*1.5:
            x2, y2 = circles[0, ind[0], :2].astype(np.int)
            mean = np.sum(obj[y2-r2:y2+r2, x2-r2:x2+r2]) / pow(r2*2,2)
            print(mean)
            if mean < 60: results['合闸指示'] = '亮'
            else: results['合闸指示'] = '灭'
        else: results['合闸指示'] = '亮'

    # 储能指示
    box1 = np.array([x-11*r, y-10*r, x-3*r, y-5*r]).astype(np.int)
    obj = img1[box1[1]:box1[3], box1[0]:box1[2]]
    cv2_show('3', obj)
    hsv = cv2.cvtColor(obj, cv2.COLOR_BGR2HSV) # 色彩空间转换为hsv，便于分离
    lower_hsv1 = np.array([26, 43, 46])  # 提取颜色的低值 yellow [26, 43, 46] 11
    high_hsv1 = np.array([34, 100, 100])  # 提取颜色的高值 [34, 255, 255]
    mask1 = cv2.inRange(hsv, lowerb=lower_hsv1, upperb=high_hsv1)
    mask=cv2.Canny(mask1, 30, 100)
    cv2_show('m2', mask)
    circles = cv2.HoughCircles(mask,cv2.HOUGH_GRADIENT,1,20,
                                param1=100,param2=20,minRadius=0,maxRadius=100) # 霍夫变换圆检测
    print(circles) #输出返回值，方便查看类型
    if circles is None: results['储能指示'] = '亮'
    else:
        ind = np.unravel_index(np.argmax(circles[0], axis=None), circles[0].shape)
        r2 = circles[0, ind[0], 2].astype(np.int)
        cv2_show('3', obj)
        if r/2 < r2 < r*1.5:
            x2, y2 = circles[0, ind[0], :2].astype(np.int)
            mean = np.sum(obj[y2-r2:y2+r2, x2-r2:x2+r2]) / pow(r2*2,2)
            print(mean)
            if mean < 60: results['储能指示'] = '亮'
            else: results['储能指示'] = '灭'
        else: results['储能指示'] = '亮'

    # 分闸指示
    box1 = np.array([x-3*r, y-10*r, x+3*r, y-4*r]).astype(np.int)
    obj = img1[box1[1]:box1[3], box1[0]:box1[2]]
    cv2_show('3', obj)
    hsv = cv2.cvtColor(obj, cv2.COLOR_BGR2HSV) # 色彩空间转换为hsv，便于分离
    lower_hsv1 = np.array([35, 100, 46]) # 提取颜色的低值 green [35, 43, 46]
    high_hsv1 = np.array([100, 255, 120]) # 提取颜色的高值 [77, 255, 255]
    mask = cv2.inRange(hsv, lowerb=lower_hsv1, upperb=high_hsv1)
    cv2_show('m2', mask)
    mask = cv2.Canny(mask,30,100)
    circles = cv2.HoughCircles(mask,cv2.HOUGH_GRADIENT,1,20,
                                param1=100,param2=20,minRadius=0,maxRadius=100) # 霍夫变换圆检测
    print(circles) #输出返回值，方便查看类型
    if circles is None: results['分闸指示'] = '亮'
    else:
        ind = np.unravel_index(np.argmax(circles[0], axis=None), circles[0].shape)
        r2 = circles[0, ind[0], 2].astype(np.int)
        cv2_show('3', obj)
        if r/2 < r2 < r*1.5:
            x2, y2 = circles[0, ind[0], :2].astype(np.int)
            mean = np.sum(obj[y2-r2:y2+r2, x2-r2:x2+r2]) / pow(r2*2,2)
            print(mean)
            if mean < 60: results['分闸指示'] = '亮'
            else: results['分闸指示'] = '灭'
        else: results['分闸指示'] = '亮'

    # 对box3检测
    box1 = np.array([x-12*r, y-22*r, x-2*r, y-13*r]).astype(np.int)
    obj = img1[box1[1]:box1[3], box1[0]:box1[2]]
    cv2_show('3', obj)
    gray_obj = cv2.cvtColor(obj, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(gray_obj, 0, 255, cv2.THRESH_OTSU) # 方法选择为THRESH_OTSU
    cv2_show('4', mask)
    X = []
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i, j] < 10:
                X.append([i, j])
    X = np.array(X)
    db = DBSCAN(eps=2, min_samples=5, metric='euclidean') # 密度聚类DBSCAN 半径，样本点数量，欧式距离
    y_db = db.fit_predict(X)
    num = {}
    for i in range(max(y_db)+1):
        num[i] = len(X[y_db==i, :])
    print(num)
    o = None
    for key,value in num.items():
        if value == max(num.values()):
            o = int(key)
    if o is not None:
        X_obj = X[y_db == o, :]
    else:
        raise ValueError
    box = np.array([min(X_obj[:, 1]), min(X_obj[:, 0]), max(X_obj[:, 1]), max(X_obj[:, 0])]) # xmin, ymin, xmax, ymax
    print(box)
    box = (box / f_xy).astype(np.int)
    box1 = (box1 / f_xy).astype(np.int)
    obj_o = img[box1[1]:box1[3], box1[0]:box1[2]]
    img_obj = obj_o[box[1]:box[3], box[0]:box[2]]
    # cv2_show('3', img_obj)
    hsv = cv2.cvtColor(img_obj, cv2.COLOR_BGR2HSV) # 色彩空间转换为hsv，便于分离
    lower_hsv1 = np.array([0, 0, 0]) # 提取颜色的低值 black [0, 0, 0]
    high_hsv1 = np.array([180, 255, 46]) # 提取颜色的高值 [180, 255, 46]
    mask = cv2.inRange(hsv, lowerb=lower_hsv1, upperb=high_hsv1)
    cv2_show('m3', mask)
    print("enc")
    return results

def Point_thirty_four(file_name1, file_name2):
    Point_info = [ '合闸指示', '储能指示', '分闸指示', '合闸按钮', '合闸压板', '分闸按钮']
    # result_imgs = img34_1(file_name2)
    # cv2_show('result', result_imgs)
    results2 = img34_2(file_name1)
    # results2['电压1'] = str(0)
    # results2['电压2'] = str(0)
    # results2['电压3'] = str(0)
    # results2['电压4'] = str(0)
    # results2['调度号'] = dispatchNum_recog(result_imgs)
    print(results2)
    Point_list = []
    for info in Point_info:
        Point_list.append(results2[info])
    print(Point_list)
    return Point_list


if __name__ == '__main__':
    img1_path = './images4/36-1.JPG'
    img2_path = './images4/36-2.JPG'
    Point_info = ['调度号', '合闸指示', '储能指示', '分闸指示', '合闸按钮', '合闸压板', '分闸按钮', '电压1', '电压2', '电压3', '电压4']
    # result_imgs = img36_1(img1_path)
    # cv2_show('result', result_imgs)
    results2 = img34_2(img2_path)
    results2['电压1'] = str(0)
    results2['电压2'] = str(0)
    results2['电压3'] = str(0)
    results2['电压4'] = str(0)
    results2['调度号'] = str(441) # dispatchNum_recog(result_imgs)
    print(results2)
    Point_list = []
    for info in Point_info:
        Point_list.append(results2[info])
    print(Point_list)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

