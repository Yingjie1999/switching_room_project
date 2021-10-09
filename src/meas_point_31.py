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

def img31_1(file_path):
    img = cv2.imread(file_path)
    f_xy = 0.1
    img1 = cv2.resize(img.copy(), None, fx=f_xy, fy=f_xy)
    hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)  # 色彩空间转换为hsv，便于分离
    lower_hsv1 = np.array([0, 0, 0])  # 提取颜色的低值 black [0, 0, 0]
    high_hsv1 = np.array([180, 255, 80])  # 提取颜色的高值 [180, 255, 46]
    mask1 = cv2.inRange(hsv, lowerb=lower_hsv1, upperb=high_hsv1)
    #cv2_show#('1', img)
    #cv2_show#('m1', mask1)
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
    #cv2_show#('o', region)
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)  # 色彩空间转换为hsv，便于分离
    lower_hsv1 = np.array([156, 43, 46]) # 提取颜色的低值 red [156, 43, 46]
    high_hsv1 = np.array([180, 255, 255]) # 提取颜色的高值 [180, 255, 255]
    lower_hsv2 = np.array([0, 43, 46]) # 提取颜色的低值 red [0, 43, 46]
    high_hsv2 = np.array([10, 255, 255]) # 提取颜色的高值 [10, 255, 255]
    mask_1 = cv2.inRange(hsv, lowerb=lower_hsv1, upperb=high_hsv1)
    mask_2 = cv2.inRange(hsv, lowerb=lower_hsv2, upperb=high_hsv2)
    mask1 = cv2.bitwise_or(mask_1, mask_2)
    #cv2_show#('m', mask1)
    X = []
    for i in range(mask1.shape[0]):
        for j in range(mask1.shape[1]):
            if mask1[i, j] > 10:
                X.append([i, j])
    X = np.array(X)
    box = np.array([min(X[:, 1]), min(X[:, 0]), max(X[:, 1]), max(X[:, 0])]) # xmin, ymin, xmax, ymax
    obj = mask1[box[1]:box[3], box[0]:box[2]]
    #cv2_show#('m2', obj)
    return obj

def img31_2(file_path):
    results = {}
    # 找到黄色显示灯
    img = cv2.imread(file_path)
    f_xy = 0.2
    img1 = cv2.resize(img.copy(), None, fx=f_xy, fy=f_xy)
    hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)  # 色彩空间转换为hsv，便于分离
    lower_hsv1 = np.array([26, 43, 46])  # 提取颜色的低值 yellow [26, 43, 46] 11
    high_hsv1 = np.array([34, 100, 100])  # 提取颜色的高值 [34, 255, 255]
    mask0 = cv2.inRange(hsv, lowerb=lower_hsv1, upperb=high_hsv1)
    mask=cv2.Canny(mask0,30,100)
    #cv2_show#('m0', mask0)
    #cv2_show#('m1', mask)
    circles = cv2.HoughCircles(mask,cv2.HOUGH_GRADIENT,1,20,
                                param1=100,param2=20,minRadius=0,maxRadius=100) # 霍夫变换圆检测
    print(circles) # 输出返回值，方便查看类型
    if circles is None:
        print('储能指示亮')
        results["储能指示"] = '亮'
        lower_hsv1 = np.array([11, 100, 100])  # 提取颜色的低值 yellow [26, 43, 46]
        high_hsv1 = np.array([34, 255, 255])  # 提取颜色的高值 [34, 255, 255]
        mask0 = cv2.inRange(hsv, lowerb=lower_hsv1, upperb=high_hsv1)
        mask = cv2.Canny(mask0, 30, 100)
        #cv2_show#('m0', mask0)
        #cv2_show#('m1', mask)
        circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, 1, 20,
                                   param1=100, param2=20, minRadius=0, maxRadius=100)  # 霍夫变换圆检测
        for circle in circles[0]:
            c = circle.astype(np.int)
            # 在原图用指定颜色圈出圆，参数设定为int所以圈画存在误差
            img1 = cv2.circle(img1, (c[0], c[1]), c[2], (0, 0, 255), 2, 8, 0)  # 坐标行列(就是圆心) 半径
        ind = np.unravel_index(np.argmax(circles[0], axis=None), circles[0].shape)
        x, y, r = circles[0, ind[0], :].astype(np.int)
        print(x, y, r)
        #cv2_show#('1', img1)
    else:
        print('储能指示灭')
        results['储能指示'] = '灭'
        for circle in circles[0]:
            c = circle.astype(np.int)
            # 在原图用指定颜色圈出圆，参数设定为int所以圈画存在误差
            img1 = cv2.circle(img1, (c[0], c[1]), c[2], (0, 0, 255), 2, 8, 0)  # 坐标行列(就是圆心) 半径
        ind = np.unravel_index(np.argmax(circles[0], axis=None), circles[0].shape)
        x, y, r = circles[0, ind[0], :].astype(np.int)
        print(x, y, r)
        #cv2_show#('1', img1)

    # 确定其它目标的位置
    box1 = np.array([x-9*r, y-3*r, x-2*r, y+9*r])
    box2 = np.array([x+2*r, y-3*r, x+9*r, y+9*r])
    box3 = np.array([x-4*r, y-11*r, x+4*r, y-5*r])
    # cv2.rectangle(img1, tuple(box1[:2]), tuple(box1[2:]), [255,255,0], thickness=2)
    # cv2.rectangle(img1, tuple(box2[:2]), tuple(box2[2:]), [255,255,0], thickness=2)
    # cv2.rectangle(img1, tuple(box3[:2]), tuple(box3[2:]), [255,255,0], thickness=2)
    #cv2_show#('2', img1)

    # 对box1检测其中的灯的情况
    obj = img1[box1[1]:box1[3], box1[0]:box1[2]]
    #cv2_show#('3', obj)
    hsv = cv2.cvtColor(obj, cv2.COLOR_BGR2HSV) # 色彩空间转换为hsv，便于分离

    lower_hsv1 = np.array([156, 100, 0]) # 提取颜色的低值 red [156, 43, 46]
    high_hsv1 = np.array([180, 255, 120]) # 提取颜色的高值 [180, 255, 255]
    lower_hsv2 = np.array([0, 100, 0]) # 提取颜色的低值 red [0, 43, 46]
    high_hsv2 = np.array([10, 255, 120]) # 提取颜色的高值 [10, 255, 255]
    mask1 = cv2.inRange(hsv, lowerb=lower_hsv1, upperb=high_hsv1)
    mask2 = cv2.inRange(hsv, lowerb=lower_hsv2, upperb=high_hsv2)
    mask = cv2.bitwise_or(mask1, mask2)
    mask=cv2.Canny(mask,30,100)
    #cv2_show#('m2', mask)
    circles = cv2.HoughCircles(mask,cv2.HOUGH_GRADIENT,1,20,
                                param1=100,param2=20,minRadius=0,maxRadius=100) # 霍夫变换圆检测
    print(circles) #输出返回值，方便查看类型
    if circles is None: results['合闸指示'] = '亮'
    else:
        ind = np.unravel_index(np.argmax(circles[0], axis=None), circles[0].shape)
        r2 = circles[0, ind[0], 2].astype(np.int)
        #cv2_show#('3', obj)
        if r/2 < r2 < r*1.5:
            x, y = circles[0, ind[0], :2].astype(np.int)
            mean = np.sum(obj[y-r2:y+r2, x-r2:x+r2]) / pow(r2*2,2)
            print(mean)
            if mean < 60: results['合闸指示'] = '亮'
            else: results['合闸指示'] = '灭'
        else: results['合闸指示'] = '亮'

    lower_hsv1 = np.array([35, 100, 46]) # 提取颜色的低值 green [35, 43, 46]
    high_hsv1 = np.array([100, 255, 120]) # 提取颜色的高值 [77, 255, 255]
    mask = cv2.inRange(hsv, lowerb=lower_hsv1, upperb=high_hsv1)
    #cv2_show#('m2', mask)
    mask=cv2.Canny(mask,30,100)
    circles = cv2.HoughCircles(mask,cv2.HOUGH_GRADIENT,1,20,
                                param1=100,param2=20,minRadius=0,maxRadius=100) # 霍夫变换圆检测
    print(circles) #输出返回值，方便查看类型
    if circles is None: results['合闸按钮'] = '亮'
    else:
        ind = np.unravel_index(np.argmax(circles[0], axis=None), circles[0].shape)
        r2 = circles[0, ind[0], 2].astype(np.int)
        #cv2_show#('3', obj)
        if r/2 < r2 < r*1.5:
            x, y = circles[0, ind[0], :2].astype(np.int)
            mean = np.sum(obj[y-r2:y+r2, x-r2:x+r2]) / pow(r2*2,2)
            print(mean)
            if mean < 60: results['合闸按钮'] = '亮'
            else: results['合闸按钮'] = '灭'
        else: results['合闸按钮'] = '亮'

    # 对box2检测其中的灯的情况
    obj = img1[box2[1]:box2[3], box2[0]:box2[2]]
    #cv2_show#('3', obj)
    hsv = cv2.cvtColor(obj, cv2.COLOR_BGR2HSV) # 色彩空间转换为hsv，便于分离

    lower_hsv1 = np.array([156, 100, 0]) # 提取颜色的低值 red [156, 43, 46]
    high_hsv1 = np.array([180, 255, 120]) # 提取颜色的高值 [180, 255, 255]
    lower_hsv2 = np.array([0, 100, 0]) # 提取颜色的低值 red [0, 43, 46]
    high_hsv2 = np.array([10, 255, 120]) # 提取颜色的高值 [10, 255, 255]
    mask1 = cv2.inRange(hsv, lowerb=lower_hsv1, upperb=high_hsv1)
    mask2 = cv2.inRange(hsv, lowerb=lower_hsv2, upperb=high_hsv2)
    mask = cv2.bitwise_or(mask1, mask2)
    mask = cv2.Canny(mask,30,100)
    #cv2_show#('m2', mask)
    circles = cv2.HoughCircles(mask,cv2.HOUGH_GRADIENT,1,20,
                                param1=100,param2=20,minRadius=0,maxRadius=100) # 霍夫变换圆检测
    print(circles) # 输出返回值，方便查看类型
    if circles is None: results['分闸按钮'] = '亮'
    else:
        ind = np.unravel_index(np.argmax(circles[0], axis=None), circles[0].shape)
        r2 = circles[0, ind[0], 2].astype(np.int)
        #cv2_show#('3', obj)
        if r/2 < r2 < r*1.5:
            x, y = circles[0, ind[0], :2].astype(np.int)
            mean = np.sum(obj[y-r2:y+r2, x-r2:x+r2]) / pow(r2*2,2)
            print(mean)
            if mean < 60: results['分闸按钮'] = '亮'
            else: results['分闸按钮'] = '灭'
        else: results['分闸按钮'] = '亮'

    lower_hsv1 = np.array([35, 100, 46]) # 提取颜色的低值 green [35, 43, 46]
    high_hsv1 = np.array([100, 255, 120]) # 提取颜色的高值 [77, 255, 255]
    mask = cv2.inRange(hsv, lowerb=lower_hsv1, upperb=high_hsv1)
    #cv2_show#('m2', mask)
    mask = cv2.Canny(mask,30,100)
    circles = cv2.HoughCircles(mask,cv2.HOUGH_GRADIENT,1,20,
                                param1=100,param2=20,minRadius=0,maxRadius=100) # 霍夫变换圆检测
    print(circles) #输出返回值，方便查看类型
    if circles is None: results['分闸指示'] = '亮'
    else:
        ind = np.unravel_index(np.argmax(circles[0], axis=None), circles[0].shape)
        r2 = circles[0, ind[0], 2].astype(np.int)
        #cv2_show#('3', obj)
        if r/2 < r2 < r*1.5:
            x, y = circles[0, ind[0], :2].astype(np.int)
            mean = np.sum(obj[y-r2:y+r2, x-r2:x+r2]) / pow(r2*2,2)
            print(mean)
            if mean < 60: results['分闸指示'] = '亮'
            else: results['分闸指示'] = '灭'
        else: results['分闸指示'] = '亮'

    # 对box3检测
    obj = img1[box3[1]:box3[3], box3[0]:box3[2]]
    #cv2_show#('3', obj)
    gray_obj = cv2.cvtColor(obj, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(gray_obj, 0, 255, cv2.THRESH_OTSU) #方法选择为THRESH_OTSU
    #cv2_show#('4', mask)
    X = []
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i, j] < 10:
                X.append([i, j])
    X = np.array(X)
    db = DBSCAN(eps=3, min_samples=5, metric='euclidean')#密度聚类DBSCAN 半径，样本点数量，欧式距离
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
    box = np.array([min(X_obj[:, 1]), min(X_obj[:, 0]), max(X_obj[:, 1]), max(X_obj[:, 0])]).astype(np.int) # xmin, ymin, xmax, ymax
    print(box)
    # box_r = (box / f_xy).astype(np.int)
    img_obj = obj[box[1]:box[3], box[0]:box[2]]
    #cv2_show#('3', img_obj)
    hsv = cv2.cvtColor(img_obj, cv2.COLOR_BGR2HSV) # 色彩空间转换为hsv，便于分离
    lower_hsv1 = np.array([0, 0, 0]) # 提取颜色的低值 black [0, 0, 0]
    high_hsv1 = np.array([180, 255, 46]) # 提取颜色的高值 [180, 255, 46]
    mask = cv2.inRange(hsv, lowerb=lower_hsv1, upperb=high_hsv1)
    #cv2_show#('m3', mask)
    return results

def Point_thirty_one(file_name1, file_name2):
    Point_info = [ '合闸指示', '储能指示', '分闸指示', '合闸按钮', '分闸按钮',]
    result_imgs = img31_1(file_name1)
    #cv2_show#('result', result_imgs)
    results2 = img31_2(file_name2)
    # results2['电压1'] = str(0)
    # results2['电压2'] = str(0)
    # results2['电压3'] = str(0)
    # results2['电压4'] = str(0)
    # results2['调度号'] = str(0)
    print(results2)

    Point_list = []
    for info in Point_info:
        Point_list.append(results2[info])
    print(Point_list)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return Point_list

if __name__ == '__main__':
    img1_path = './images2/31-1.JPG'
    result = img31_1(img1_path)
    #cv2_show#('result', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
