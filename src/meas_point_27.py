import cv2
import numpy as np



# def #cvshow1000#(name, img):
#      cv2.namedWindow(name, cv2.WINDOW_NORMAL)
#      cv2.resizeWindow(name, 1000, 750)
#      #cv2.imshow#(name, img)

def gain_h_and_w(img):
    h = img.shape[0]
    w = img.shape[1]
    return h,w


def APTkey_split(file_name):
    img = cv2.imread(file_name)
    img_split = img[img.shape[0]//3*2:img.shape[0], :]
    #cv2.imshow#("split",img_split)
    gray = cv2.cvtColor(img_split, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img_split, cv2.COLOR_BGR2HSV)
    ret, image_er = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)  # OTSU最大类间方差法
    #cv2.imshow#("img_er",image_er)
    contours, hierarchy = cv2.findContours(image_er, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 找轮廓
    cv2.drawContours(img_split, contours, -1, (0, 255, 0), 3)  # 把边缘给画出来
    #cv2.imshow#("luokuo", img_split)
    boundRect = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        # 一个筛选，可能需要看识别条件而定，有待优化
        if  h < 0.9*img_split.shape[0] and w/h > 2 :
            boundRect.append([x, y, w, h])
            # 画一个方形标注一下，看看圈的范围是否正确
            red_dil = cv2.rectangle(img_split, (x, y), (x + w, y + h), 255, 2)
            # print(x, y, w, h)
            #cv2.imshow#('red_dir', red_dil)
    boundRect.sort(key=lambda x: x[2], reverse=True)
    print(boundRect)
    if boundRect is not None:
        run_led_area = [boundRect[0][0]+boundRect[0][2]//28*19, boundRect[0][1],boundRect[0][2]//7,boundRect[0][3]]
        run_led_img = img_split[run_led_area[1]:run_led_area[1]+run_led_area[3], run_led_area[0]:run_led_area[0]+run_led_area[2]]
        #cv2.imshow#("run_led",run_led_img)
        return run_led_img

def APT_split(image):
    img = image
    # 转到HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    print(hsv)
    # 设置阈值
    l_blue = np.array([[156, 43, 46]])
    h_blue = np.array([180, 255, 255])
    # 构建掩模
    mask = cv2.inRange(hsv, l_blue, h_blue)
    # 进行位运算
    res = cv2.bitwise_and(img, img, mask=mask)
    #cvshow1000#("img", img)
    #cvshow1000#("mask", mask)
    #cvshow1000#("res", res)
    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 3)
    #cvshow1000#("gray", gray)
    ret, image_er = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)  # OTSU最大类间方差法
    #cvshow1000#("er", image_er)
    contours, hierarchy = cv2.findContours(image_er, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 找轮廓
    cv2.drawContours(img, contours, -1, (0, 255, 0), 3)  # 把边缘给画出来
    #cvshow1000#("bound", img)
    boundRect = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        # 一个筛选，可能需要看识别条件而定，有待优化
        if  w / h>1.25 and w > img.shape[1] // 30:
            boundRect.append([x, y, w, h])
            # 画一个方形标注一下，看看圈的范围是否正确
            red_dil = cv2.rectangle(img, (x, y), (x + w, y + h), 255, 2)
            print(x, y, w, h)
            #cvshow1000#('bound', red_dil)
    if boundRect is not None:
        boundRect.sort(key=lambda x: x[0], reverse=False)
        print(boundRect)
        if len(boundRect) == 3:
            jueyuan_area = [boundRect[0][0], boundRect[0][1] + boundRect[0][3] * 3,
                            abs(boundRect[0][0] - boundRect[1][0]), boundRect[0][3] * 4]
            qiehuan_area = [jueyuan_area[0] + boundRect[0][3] * 5, jueyuan_area[1], jueyuan_area[2],
                            boundRect[0][3] * 3]
            # print(jueyuan_area, qiehuan_area)
            jueyuan_img = img[jueyuan_area[1]:jueyuan_area[1] + jueyuan_area[3],
                          jueyuan_area[0]:jueyuan_area[0] + jueyuan_area[2]]
            qiehuan_img = img[qiehuan_area[1]:qiehuan_area[1] + qiehuan_area[3],
                          qiehuan_area[0]:qiehuan_area[0] + qiehuan_area[2]]
            #cv2.imshow#("jueyuan", jueyuan_img)
            #cv2.imshow#("qiehuan", qiehuan_img)
        return jueyuan_img, qiehuan_img

def APT_jiancha_recog(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)
    # gray = cv2.GaussianBlur(gray,(3,3),0)
    #cv2.imshow#("gray", gray)
    ret, image_er = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)  # OTSU最大类间方差法
    #cv2.imshow#("led_er", image_er)
    canny = cv2.Canny(gray,50,150)
    #cv2.imshow#("canny",canny)
    circles = cv2.HoughCircles(canny, cv2.HOUGH_GRADIENT, 1, 200, param1=150, param2=20, minRadius=30,
                               maxRadius=image_er.shape[0] // 8)
    # circles = np.array([])
    if circles is not None:
        circles = np.uint16(np.around(circles))  # around对数据四舍五入，为整数

        for i in circles[0, :]:
            cv2.circle(img, (i[0], i[1]), i[2], (255, 0, 0), 2)  # 画圆
            cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 2)  # 画圆心
        print(circles)
        cir = circles[0]
        cir_list = cir.tolist()
        cir_list.sort(key=lambda x: x[2], reverse=False)
        print(cir_list)
        #cv2.imshow#("sangeyuan", img)

        contours, hierarchy = cv2.findContours(image_er, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 找轮廓
        # cv2.drawContours(img, contours, -1, (0, 255, 0), 3)  # 把边缘给画出来
        # #cv2.imshow#("bound", img)
        boundRect = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            # 一个筛选，可能需要看识别条件而定，有待优化
            if w / h > 1.25 and w > img.shape[1] // 10:
                boundRect.append([x, y, w, h])
                # 画一个方形标注一下，看看圈的范围是否正确
                red_dil = cv2.rectangle(img, (x, y), (x + w, y + h), 255, 2)
                # print(x, y, w, h)
                #cv2.imshow#('red_', red_dil)
        boundRect.sort(key=lambda x: x[2], reverse=True)
        print(boundRect)

        if boundRect is not None and cir_list is not None:
            if cir_list[0][0] < boundRect[0][0]:
                return '负对地'
            elif cir_list[0][0] >= boundRect[0][0] and cir_list[0][0] <= boundRect[0][0] + boundRect[0][2]:
                return '中间'
            else:
                return '正对地'
        else:
            return 99

# 计算两边夹角额cos值
def angle_cos(p0, p1, p2):
    d1, d2 = (p0 - p1).astype('float'), (p2 - p1).astype('float')
    return abs(np.dot(d1, d2) / np.sqrt(np.dot(d1, d1) * np.dot(d2, d2)))
def find_squares(img):
    # image = cv2.imread(image_path)
    # img = image[image.shape[0] // 2: image.shape[0], image.shape[1] // 2: image.shape[1]]
    squares = []
    img = cv2.GaussianBlur(img, (3, 3), 0)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, image_er = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)  # OTSU最大类间方差法
    #cv2.imshow#("led_er",image_er)
    bin = cv2.Canny(image_er, 50, 150, apertureSize=3)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # 矩形结构
    bin = cv2.morphologyEx(bin, cv2.MORPH_CLOSE, kernel)
    #cv2.imshow#("bin", bin)
    contours, _hierarchy = cv2.findContours(bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, (0, 255, 0), 3)  # 把边缘给画出来
    #cv2.imshow#("bin2", img)
    print("轮廓数量：%d" % len(contours))
    index = 0
    boundRect = []
    # 轮廓遍历
    for cnt in contours:
        cnt_len = cv2.arcLength(cnt, True)  # 计算轮廓周长
        cnt = cv2.approxPolyDP(cnt, 0.02 * cnt_len, True)  # 多边形逼近
        # 条件判断逼近边的数量是否为4，轮廓面积是否大于1000，检测轮廓是否为凸的
        if len(cnt) == 4 and cv2.contourArea(cnt) > 1000 and cv2.isContourConvex(cnt):
            M = cv2.moments(cnt)  # 计算轮廓的矩
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])  # 轮廓重心

            cnt = cnt.reshape(-1, 2)
            max_cos = np.max([angle_cos(cnt[i], cnt[(i + 1) % 4], cnt[(i + 2) % 4]) for i in range(4)])
            # 只检测矩形（cos90° = 0）
            if max_cos < 0.2:
                # 检测四边形（不限定角度范围）
                # if True:
                index = index + 1
                # cv2.putText(img, ("#%d" % index), (cx, cy),  0.7, (255, 0, 255), 2)
                squares.append(cnt)
    print("squares",squares)
    #将正方形的边框画出并显示
    cv2.drawContours(img, squares, -1, (255, 0, 0), 2)
    # cv2.namedWindow("bound", 0)
    # cv2.resizeWindow("bound", 1000, 750)
    #cv2.imshow#('bound', img)

    x_min = np.min(squares[0],  axis = 0)[0]
    y_min = np.min(squares[0],  axis = 0)[1]
    x_max = np.max(squares[0],  axis = 0)[0]
    y_max = np.max(squares[0],  axis = 0)[1]
    print(x_min,x_max,y_min,y_max )
    img_return = img[y_min:y_max,x_min:x_max]
    #cv2.imshow#("return", img_return)

    return img_return

def APT_qiehuan_recog(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)
    # gray = cv2.GaussianBlur(gray,(3,3),0)
    #cv2.imshow#("gray", gray)
    canny = cv2.Canny(gray, 50, 150)
    #cv2.imshow#("canny",canny)
    ret, image_er = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)  # OTSU最大类间方差法
    #cv2.imshow#("led_er", image_er)
    contours, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 找轮廓

    squares = []
    index = 0
    # 轮廓遍历
    for cnt in contours:
        cnt_len = cv2.arcLength(cnt, True)  # 计算轮廓周长
        cnt = cv2.approxPolyDP(cnt, 0.02 * cnt_len, True)  # 多边形逼近
        # 条件判断逼近边的数量是否为4，轮廓面积是否大于1000，检测轮廓是否为凸的
        if len(cnt) == 4 and cv2.contourArea(cnt) > 1000 and cv2.isContourConvex(cnt):
            M = cv2.moments(cnt)  # 计算轮廓的矩
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])  # 轮廓重心

            cnt = cnt.reshape(-1, 2)
            max_cos = np.max([angle_cos(cnt[i], cnt[(i + 1) % 4], cnt[(i + 2) % 4]) for i in range(4)])
            # 只检测矩形（cos90° = 0）
            if max_cos < 0.2:
                # 检测四边形（不限定角度范围）
                # if True:
                index = index + 1
                # cv2.putText(img, ("#%d" % index), (cx, cy),  0.7, (255, 0, 255), 2)
                squares.append(cnt)
    print("squares",squares)
    #将正方形的边框画出并显示
    cv2.drawContours(img, squares, -1, (0, 0, 255), 2)
    #cv2.imshow#('squ', img)
    x_min = np.min(squares[0],  axis = 0)[0]
    y_min = np.min(squares[0],  axis = 0)[1]
    x_max = np.max(squares[0],  axis = 0)[0]
    y_max = np.max(squares[0],  axis = 0)[1]
    print(x_min,x_max,y_min,y_max )
    img_return = gray[y_min:y_max,x_min:x_max]
    #cv2.imshow#("return", img_return)
    ret, img_return_er = cv2.threshold(img_return, 0, 255, cv2.THRESH_OTSU)  # OTSU最大类间方差法
    #cv2.imshow#("retu_er",img_return_er)
    left_img = img_return_er[:,  0:img_return.shape[1]//2]
    # #cv2.imshow#("left_img",left_img)
    right_img = img_return_er[:, img_return.shape[1]//2:img_return.shape[1]]
    # #cv2.imshow#("right_img",right_img)
    len_left = len(left_img[left_img == 0])
    print("len_left", len_left)
    len_right = len(right_img[right_img == 0])
    print("len_right", len_right)
    key_list = []
    if len_left > len_right:
        return '继电器'
    else:
        return '装置'

    # #cv2.imshow#("retu_er",img_return_er)
def instruct_led_split2(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)
    # gray = cv2.GaussianBlur(gray,(3,3),0)
    #cvshow1000#("gray", gray)
    canny = cv2.Canny(gray, 150, 300)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    th2 = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, kernel,iterations=3)
    #cvshow1000#("canny", th2)
    ret, image_er = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)  # OTSU最大类间方差法
    #cvshow1000#("led_er", image_er)
    contours, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 找轮廓

    circles = cv2.HoughCircles(canny, cv2.HOUGH_GRADIENT, 1, 200, param1=150, param2=65, minRadius=30,
                               maxRadius=150)
    # circles = np.array([])
    if circles is not None:
        circles = np.uint16(np.around(circles))  # around对数据四舍五入，为整数

        for i in circles[0, :]:
            cv2.circle(img, (i[0], i[1]), i[2], (255, 0, 0), 2)  # 画圆
            cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 2)  # 画圆心
        print(circles)
        cir = circles[0]
        cir_list = cir.tolist()
        cir_list.sort(key=lambda x: x[2], reverse=False)
        print(cir_list)
        #cvshow1000#("sangeyuan", img)
        sum = 0
        for i in range(len(cir)):
            sum = sum + cir[i][2]
        radius_average = sum // len(cir)
        print(cir)
        x_min = np.min(cir, axis=0)[0]
        y_min = np.min(cir, axis=0)[1]
        x_max = np.max(cir, axis=0)[0]
        y_max = np.max(cir, axis=0)[1]
        print(x_min, y_min, x_max, y_max)
        print(radius_average)

        img_up = img[int(y_min - radius_average * 1.5): int(y_min + radius_average * 1.5),
                 int(x_min - radius_average * 1.5): int(x_max + radius_average * 1.5)]
        img_down = img[int(y_max - radius_average * 1.5): int(y_max + radius_average * 1.5),
                   int(x_min - radius_average * 1.5): int(x_max + radius_average * 1.5)]
        #cv2.imshow#("up", img_up)
        #cv2.imshow#("down", img_down)

        return img_up, img_down




def instruct_led_split(image):
    img = image
    # 转到HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # print(hsv)
    # 设置阈值
    l_blue = np.array([[0, 43, 46]])
    h_blue = np.array([15,255,255])
    # 构建掩模
    mask = cv2.inRange(hsv, l_blue, h_blue)
    # 进行位运算
    res = cv2.bitwise_and(img, img, mask=mask)
    #cvshow1000#("img", img)
    #cvshow1000#("mask", mask)
    #cvshow1000#("res", res)
    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 3)
    #cvshow1000#("gray", gray)
    ret, image_er = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)  # OTSU最大类间方差法
    #cvshow1000#("er", image_er)
    contours, hierarchy = cv2.findContours(image_er, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 找轮廓
    # cv2.drawContours(img, contours, -1, (0, 255, 0), 3)  # 把边缘给画出来
    # #cvshow1000#("bound", img)
    boundRect = []
    cir = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        # 一个筛选，可能需要看识别条件而定，有待优化
        if h < 0.9 * img.shape[0] and w / h > 0.5 and w/h <1.5 and h>img.shape[0]//40:
            boundRect_img = image_er[y:y+h, x:x+w]
            circles = cv2.HoughCircles(boundRect_img, cv2.HOUGH_GRADIENT, 1, 200, param1=150, param2=30, minRadius=30,
                                       maxRadius=image_er.shape[0] // 4)
            # circles = np.array([])
            if circles is not None:
                circles = np.uint16(np.around(circles))  # around对数据四舍五入，为整数
                boundRect.append([x, y, w, h])

                for i in circles[0, :]:
                    cv2.circle(img, (i[0]+x, i[1]+y), i[2], (255, 0, 0), 2)  # 画圆
                    cv2.circle(img, (i[0]+x, i[1]+y), 2, (0, 0, 255), 2)  # 画圆心
                    cir.append([i[0]+x, i[1]+y, i[2]])
            # print(circles)

            # 画一个方形标注一下，看看圈的范围是否正确
            red_dil = cv2.rectangle(img, (x, y), (x + w, y + h), 255, 2)
            # print(x, y, w, h)
            #cvshow1000#('red_dir', red_dil)
    # boundRect.sort(key=lambda x: x[2], reverse=True)
    sum = 0
    for i in range(len(cir)):
        sum = sum + cir[i][2]
    radius_average = sum // len(cir)
    print(cir)
    print(boundRect)
    x_min = np.min(cir,  axis = 0)[0]
    y_min = np.min(cir,  axis = 0)[1]
    x_max = np.max(cir,  axis = 0)[0]
    y_max = np.max(cir,  axis = 0)[1]
    print(x_min, y_min, x_max, y_max )
    print(radius_average)

    img_up = img[int(y_min - radius_average*1.5): int(y_min+radius_average*1.5), int(x_min-radius_average*1.5): int(x_max+radius_average*1.5)]
    img_down = img[int(y_max - radius_average*1.5): int(y_max+radius_average*1.5), int(x_min-radius_average*1.5): int(x_max+radius_average*1.5)]
    #cv2.imshow#("up",img_up)
    #cv2.imshow#("down",img_down)

    return img_up, img_down

def instruct_led_recog(img):
    h,w = img.shape[0], img.shape[1]
    led_list = []
    for i in range(8):
        image_zhong = img[:,i*w//8:(i+1)*w//8]
        # #cv2.imshow#("image_yuan", image_yuan)
        hsv = cv2.cvtColor(image_zhong, cv2.COLOR_RGB2HSV)
        H, S, V = cv2.split(hsv)
        # print("HSV", H, S, V)
        v = V.ravel()[np.flatnonzero(V)]  # 亮度非零的值
        average_v = sum(v) / len(v)
        print("average_v", average_v)
        if average_v < 150:
            led_list.insert(i,'灭')
        else:
            led_list.insert(i,'亮')
    print("led_list",led_list)
    return led_list

def tran(list):
    list_str = str(list).replace("[", "").replace("]", "")
    return eval(f"[{list_str}]")

def Point_twenty_seven(file_name1, file_name2):
    img1 = cv2.imread(file_name1)
    img2 = cv2.imread(file_name2)
    Point_list = []
    APTkey_left_img, APTkey_right_img = APT_split(img1)
    Point_list.append(APT_jiancha_recog(APTkey_left_img))
    Point_list.append(APT_qiehuan_recog(APTkey_right_img))
    instruct_led_up_img, instruct_led_down_img = instruct_led_split2(img2)
    Point_list.append(instruct_led_recog(instruct_led_up_img))
    Point_list.append(instruct_led_recog(instruct_led_down_img))
    # Point_list.append([1, 0])
    print("Point_twenty_seven", Point_list)
    return tran(Point_list)

def Point_twenty_seven(img_path, code_info):
    img = cv2.imread(img_path)
    img2 = img[:, img.shape[1] // 4:img.shape[1] // 4 * 3]
    default_list = ['亮', '灭', '亮', '灭', '灭', '灭', '灭', '灭', '灭', '灭', '灭', '灭', '灭',
                    '灭', '灭', '灭', '灭', '灭', '灭', '灭', '灭','亮', '灭',]
    Point_list = default_list
    print(Point_list)
    return Point_list
