import cv2
import os
import numpy as np
import pytesseract
import recognition
import split
from repetition import Repetition
import handcart


#TODO:将一个测点的所有的函数封装到一个类中
def cvshow800(name, img):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, 800, 900)
    cv2.imshow(name, img)


def digit_num_split(image):
    img = image[0:image.shape[0]//2, :]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray , (3,3) , 0)
    ret, image_er = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)  # OTSU最大类间方差法
    # cv2.imshow("er",image_er)
    contours, hierarchy = cv2.findContours(image_er, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 找轮廓
    # cv2.drawContours(img, contours, -1, (0, 255, 0), 3)  # 把边缘给画出来
    # cv2.imshow("bound", img)
    boundRect = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        # 一个筛选，可能需要看识别条件而定，有待优化
        if w / h < 1.5  and w / h > 0.5and  h < 0.9 * img.shape[0] and h > 0.1 * img.shape[0]:
            boundRect.append([x, y, w, h])
            # 画一个方形标注一下，看看圈的范围是否正确
            red_dil = cv2.rectangle(img, (x, y), (x + w, y + h), 255, 2)
            print(x, y, w, h)
            # cv2.namedWindow("bound", 0)
            # cv2.resizeWindow("bound", 1000, 750)
            # cv2.imshow('red', red_dil)
    if boundRect is not None:
        print(boundRect)
        boundRect.sort(key=lambda x: x[2], reverse=False)
        img_return = img[boundRect[0][1]:boundRect[0][1] + boundRect[0][3],
                     boundRect[0][0]:boundRect[0][0] + boundRect[0][2]]
        cv2.imshow("digit_img", img_return)
        return img_return





def dispatchNum_split(img):
    image = img
    split_h = image.shape[0]
    split_w = image.shape[1]
    image_split = image[split_h // 4:split_h // 4 * 3, split_w // 4:3 * split_w // 4]
    # cv2.imshow("image_split",image_split)
    gray = cv2.cvtColor(image_split, cv2.COLOR_BGR2GRAY)
    # cv2.namedWindow("gray",cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("gray",1000,750)
    # cv2.imshow("gray",gray)

    image_g = cv2.GaussianBlur(gray, (3, 3), 0)
    # cv2.namedWindow("gauss", 0)
    # cv2.resizeWindow("gauss", 1000, 750)
    # cv2.imshow("gauss", image_g)

    ret, image_er = cv2.threshold(image_g, 0, 255, cv2.THRESH_OTSU)  # OTSU最大类间方差法
    # cv2.namedWindow("er", cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("er", 1000, 750)
    # cv2.imshow("er", image_er)

    contours, hierarchy = cv2.findContours(image_er, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 找轮廓
    # cv2.drawContours(image_split, contours, -1, (0, 255, 0), 3)  # 把边缘给画出来
    # cv2.namedWindow("luokuo",0)
    # cv2.resizeWindow("luokuo",1000,750)
    # cv2.imshow("luokuo",image_split)
    boundRect = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        # 一个筛选，可能需要看识别条件而定，有待优化
        if w / h > 1.5 and w < 0.9 * split_w and w > 0.1 * split_w and h < 0.9 * split_h and h > 0.1 * split_h:
            boundRect.append([x, y, w, h])
            # 画一个方形标注一下，看看圈的范围是否正确
            # red_dil = cv2.rectangle(image_split, (x, y), (x + w, y + h), 255, 2)
            # print(x, y, w, h)
            # cv2.namedWindow("bound", 0)
            # cv2.resizeWindow("bound", 1000, 750)
            # cv2.imshow('bound', red_dil)
    print(boundRect)
    # print(np.shape(boundRect))
    # 暂时通过最大值来判断
    a = np.array(boundRect)
    maxindex = a.argmax(axis=0)
    print(maxindex)
    black = []
    black = (boundRect[maxindex[2]])
    print(black)

    # dispatchNum_coordinate = []
    dispatchNum_coordinate_x = black[0] + black[2] // 3
    dispatchNum_coordinate_y = black[1] - black[3] // 2
    dispatchNum_coordinate_w = black[2] // 3
    dispatchNum_coordinate_h = black[3] // 2
    dispatchNum_coordinate = np.array(
        [dispatchNum_coordinate_x, dispatchNum_coordinate_y, dispatchNum_coordinate_w, dispatchNum_coordinate_h])
    image_out = image_split[dispatchNum_coordinate_y:dispatchNum_coordinate_y + dispatchNum_coordinate_h,
                dispatchNum_coordinate_x:dispatchNum_coordinate_x + dispatchNum_coordinate_w]
    # cv2.imshow("image_out",image_out)
    print(dispatchNum_coordinate)
    diaoduhao40 = cv2.rectangle(image_split, (dispatchNum_coordinate_x, dispatchNum_coordinate_y), (dispatchNum_coordinate_x + dispatchNum_coordinate_w,
                                                                                                dispatchNum_coordinate_y + dispatchNum_coordinate_h), (0,0,255), 2)
    cv2.namedWindow("40",0)
    cv2.resizeWindow("40",1000,750)
    cv2.imshow("40",diaoduhao40)
    return image_out

def dispatchNum_recog(img):
    txt = recognition.serial_recogn(img)
    print(txt)
    return txt


#这个函数用来将L1、L2、L3的部分给分割出来
def L_led_split(img):
    image = img
    split_h = image.shape[0] // 2
    split_w = image.shape[1] // 2
    # print("split_h:", split_h)
    # print("split_w:", split_w)
    image_split = image[split_h: split_h * 2, 0: split_w]
    gray = cv2.cvtColor(image_split, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("gray", gray)
    ret, image_er = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)  # OTSU最大类间方差法
    # cv2.namedWindow("er", cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("er", 1000, 750)
    cv2.imshow("er", image_er)
    contours, hierarchy = cv2.findContours(image_er, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 找轮廓
    # cv2.drawContours(image_split, contours, -1, (0, 255, 0), 3)  # 把边缘给画出来
    # cv2.namedWindow("luokuo",0)
    # cv2.resizeWindow("luokuo",1000,750)
    # cv2.imshow("luokuo",image_split)
    boundRect = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        # 一个筛选，可能需要看识别条件而定，有待优化
        if w / h > 1 and w < 0.8 * split_w and w > 0.1 * split_w and h < 0.8 * split_h and h > 0.1 * split_h:
            boundRect.append([x, y, w, h])
            # 画一个方形标注一下，看看圈的范围是否正确
            # red_dil = cv2.rectangle(image_split, (x, y), (x + w, y + h), 255, 2)
            # # print(x, y, w, h)
            # cv2.namedWindow("bound", 0)
            # cv2.resizeWindow("bound", 1000, 750)
            # cv2.imshow('bound', red_dil)
    # print(boundRect)
    # print(np.shape(boundRect))
    # 暂时通过最大值来判断
    a = np.array(boundRect)
    maxindex = a.argmax(axis=0)
    # print(maxindex)
    led_bound = []
    led_bound = (boundRect[maxindex[2]])
    print(led_bound)
    led_coordinate_x = led_bound[0]
    led_coordinate_y = led_bound[1]
    led_coordinate_w = led_bound[2]
    led_coordinate_h = led_bound[3] // 2
    led_coordinate = np.array([led_coordinate_x, led_coordinate_y, led_coordinate_w, led_coordinate_h])
    image_out = image_split[led_coordinate_y:led_coordinate_h + led_coordinate_y,
                            led_coordinate_x: led_coordinate_w + led_coordinate_x]
    return image_out, led_coordinate

#通过这个函数可以直接获得开关的开关状态，1表示打向右边（开），0表示打向左边（关）
def b_w_key_split(img, L_led_area):
    image = img
    image_split = image[image.shape[0]//2+L_led_area[1]:image.shape[0]//2+L_led_area[1]+L_led_area[3]*3,
                  L_led_area[0]+L_led_area[2]//4*9:L_led_area[0]+L_led_area[2]*3]
    cv2.imshow("img_split", image_split)
    gray = cv2.cvtColor(image_split, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)
    # cv2.imshow("gray", gray)
    ret, image_er = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)  # OTSU最大类间方差法
    cv2.imshow("er", image_er)
    contours, hierarchy = cv2.findContours(image_er, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 找轮廓
    # cv2.drawContours(image_split, contours, -1, (0, 255, 0), 3)  # 把边缘给画出来

    circles = cv2.HoughCircles(image_er, cv2.HOUGH_GRADIENT, 1, 300, param1=300, param2=25, minRadius=10,
                               maxRadius=100)                                             #不同测点对于圆的判定阈值会不同，可以尝试 自适应阈值，把阴影个去掉，还有就是对于圆检测再做尝试

    circles1 = np.array([])
    if circles is not None:
        circles = np.uint16(np.around(circles))  # around对数据四舍五入，为整数
        leng = len(circles[0])
        circles1 = circles[0]
        for i in circles[0, :]:
            # cv2.circle(image_split, (i[0], i[1]), i[2], (0, 0, 255), 2)  # 画圆
            cv2.circle(image_split, (i[0], i[1]), 2, (0, 0, 255), 2)  # 画圆心
        print(circles)
        cv2.imshow("cir", image_split)

        img_return_area = [circles1[0][0] - circles1[0][2] * 2 +  L_led_area[0]+L_led_area[2]//4*9, circles1[0][1] - circles1[0][2] * 2 + image.shape[0]//2+L_led_area[1],
                           circles1[0][2] * 4, circles1[0][2] * 4]
        image_zhong = image_split[circles1[0][1] - circles1[0][2]:circles1[0][1] + circles1[0][2],
                      circles1[0][0] - circles1[0][2]:circles1[0][0] + circles1[0][2]]

        # image_yuan = gray[circles1[0][1] - circles1[0][2]:circles1[0][1] + circles1[0][2],
        #              circles1[0][0] - circles1[0][2]:circles1[0][0] + circles1[0][2]]
        # cv2.imshow("image_yuan",image_yuan)
        cv2.imshow("b_w_img", image_zhong)
        return image_zhong, img_return_area
    else:
        return 0,0


def b_w_key_recog(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    edge = cv2.Canny(gray, 50, 150)
    # cv2.imshow("bound", edge)
    lines = cv2.HoughLinesP(edge, 1, np.pi / 180, threshold=30, minLineLength=15)
    for x1, y1, x2, y2 in lines[:, 0]:
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    print("line", lines)
    cv2.imshow("xian,",img)
    k = []
    for x1, y1, x2, y2 in lines[:, 0]:
        k.append((y1 - y2) / (x1 - x2))
    k_sum = sum(k)
    if k_sum < -1 / 5:
        return '开'
    else:
        return '关'



# 计算两边夹角额cos值
def angle_cos(p0, p1, p2):
    d1, d2 = (p0 - p1).astype('float'), (p2 - p1).astype('float')
    return abs(np.dot(d1, d2) / np.sqrt(np.dot(d1, d1) * np.dot(d2, d2)))
def find_squares(img):
    image = img
    img = image[image.shape[0] // 2: image.shape[0], image.shape[1] // 2: image.shape[1]]
    squares = []
    img = cv2.GaussianBlur(img, (3, 3), 0)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bin = cv2.Canny(gray, 100, 200, apertureSize=3)
    cv2.namedWindow("canny",0)
    cv2.resizeWindow("canny",1000,750)
    cv2.imshow("canny",bin)
    contours, _hierarchy = cv2.findContours(bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
            if max_cos < 0.1:
                # 检测四边形（不限定角度范围）
                # if True:
                index = index + 1
                # cv2.putText(img, ("#%d" % index), (cx, cy),  0.7, (255, 0, 255), 2)
                squares.append(cnt)
    print("squares",squares)
    #将正方形的边框画出并显示
    # cv2.drawContours(img, squares, -1, (255, 0, 0), 2)
    # cv2.namedWindow("bound", 0)
    # cv2.resizeWindow("bound", 1000, 750)
    # cv2.imshow('bound', img)

    x_min = np.min(squares[0],  axis = 0)[0]
    y_min = np.min(squares[0],  axis = 0)[1]
    x_max = np.max(squares[0],  axis = 0)[0]
    y_max = np.max(squares[0],  axis = 0)[1]
    print(x_min,x_max,y_min,y_max )
    img_return = img[y_min:y_max,x_min:x_max]
    cv2.imshow("return", img_return)

    return img_return

#这个是识别函数，主要将分割出来的L1、2、3的亮灭状态给区分出来，1表示亮，0表示灭
def L_led_recog(image):
    # cv2.imshow("image", image)
    # print("image_w:",image.shape[1],"image_h:",image.shape[0])
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)
    # cv2.imshow("gray", gray)
    ret, image_er = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)  # OTSU最大类间方差法
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # image_er = cv2.morphologyEx(image_er, cv2.MORPH_OPEN, kernel,iterations=3)
    # cv2.imshow("image_er",image_er)
    # 这里 灭的灯识别不出来
    # print(image_er.shape[0])
    canny = cv2.Canny(image_er, 50 , 150)
    circles = cv2.HoughCircles(canny, cv2.HOUGH_GRADIENT, 1, 35, param1=300, param2=30, minRadius=0,
                               maxRadius=image_er.shape[0] // 2)
    # print(circles)
    led_list = ['灭', '灭', '灭']
    if circles is not None:
        circles = np.uint16(np.around(circles))  # around对数据四舍五入，为整数
        for i in circles[0, :]:
            cv2.circle(image, (i[0], i[1]), i[2], (0, 0, 255), 2)  # 画圆
            cv2.circle(image, (i[0], i[1]), 2, (0, 0, 255), 2)  # 画圆心
        print(circles)
        # print(circles[0][0][0])
        # print(image.shape[1]//5*2)

        for i in range(len(circles[0])):
            image_zhong = image[circles[0][i][1] - circles[0][i][2]:circles[0][i][1] + circles[0][i][2],
                          circles[0][i][0] - circles[0][i][2]:circles[0][i][0] + circles[0][i][2]]
            # cv2.imshow("image_yuan", image_yuan)
            hsv = cv2.cvtColor(image_zhong, cv2.COLOR_RGB2HSV)
            H, S, V = cv2.split(hsv)
            # print("HSV", H, S, V)
            v = V.ravel()[np.flatnonzero(V)]  # 亮度非零的值
            average_v = sum(v) / len(v)
            print("average_v", average_v)
            # if average_v < 120:
            #     led_list.insert(i, 0)
            # else:
            #     led_list.insert(i, 1)
            if circles[0][i][0] <= image.shape[1] // 5 * 2 and average_v > 120:
                led_list[0] = '亮'

            if circles[0][i][0] > image.shape[1] // 5 * 2 and circles[0][i][0] < image.shape[1] // 5 * 3 and average_v > 120 :
                led_list[1] = '亮'

            if circles[0][i][0] >= image.shape[1] // 5 * 3 and average_v > 120:
                led_list[2] = '亮'

        print(led_list)
        cv2.imshow("yuan", image)
        return led_list

#手车位置开合闸指示灯的状态
def handcar_recog(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("gray", gray)
    ret, image_er = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)  # OTSU最大类间方差法
    contours, hierarchy = cv2.findContours(image_er, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 找轮廓
    boundRect = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        # 一个筛选，可能需要看识别条件而定，有待优化
        if h > img.shape[0] // 3 and h < img.shape[0] * 0.95 and w / h < 1.5:
            boundRect.append([x, y, w, h])
            # 画一个方形标注一下，看看圈的范围是否正确
            red_dil = cv2.rectangle(img, (x, y), (x + w, y + h), 255, 2)
            cv2.imshow("red",red_dil)
    print(boundRect)
    # cv2.imshow("min", img)
    boundRect.sort(key=lambda x: x[0], reverse=False)
    boundRect_ = boundRect
    color = []
    for i in range(len(boundRect_)):
        img_quar = img[boundRect_[i][1]:boundRect_[i][1] + boundRect_[i][3],
                   boundRect_[i][0]:boundRect_[i][0] + boundRect_[i][2]]
        hsv = cv2.cvtColor(img_quar, cv2.COLOR_BGR2HSV)
        # H,S,V = np.split(hsv)
        hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
        object_H = np.argmax(hist)
        print("object_H:", object_H)

        if object_H > 156 and object_H < 180:
            color.append("合闸")
        elif object_H > 0 and object_H < 10:
            color.append("合闸")
        elif object_H > 10 and object_H < 34:
            color.append("分闸")
        else:
            color.append(" ")
        print("color:", color)
    return color

def handcar_split(img, last_area):
    h,w = img.shape[0], img.shape[1]
    handcar_location =[last_area[0], last_area[1]-last_area[3]//2*3, last_area[2], last_area[3]//4*3]
    print("handcar_",handcar_location)
    led_img = img[handcar_location[1]:handcar_location[1]+handcar_location[3]*4//3,handcar_location[0]:handcar_location[0]+handcar_location[2]]
    cv2.imshow("hand_img",led_img)
    return led_img


def digit_recog(digit_image):
    digit_list = []
    for i in range(len(digit_image) ):
        # cv2.imshow('obj_digit{}'.format(i),digit_image[i])
        digit_list.append(recognition.digit_detect(digit_image[i]))
    print(digit_list)
    return digit_list

def tran(list):
    list_str = str(list).replace("[", "").replace("]", "")
    return eval(f"[{list_str}]")


# def Point_one(file_name2, code_info):
#     # img1 = cv2.imread(file_name1)
#     img2 = cv2.imread(file_name2)
#     img2 = img2[:, img2.shape[1]//4:img2.shape[1]//4*3]
#     # cvshow800("input_split", img2)
#     Point_list = []
#     # # num_img = dispatchNum_split(img1)
#     # # Point_list.append(dispatchNum_recog(num_img))
#     L_led_img, L_led_area = L_led_split(img2)
#     Point_list.append(L_led_recog(L_led_img))
#     b_w_key_img, b_w_key_area = b_w_key_split(img2, L_led_area)
#     Point_list.append(b_w_key_recog(b_w_key_img))
#     handcar_img = handcar_split(img2, b_w_key_area)
#     Point_list.append(handcar_recog(handcar_img))
#     # digit_image = split.detect_and_split_1(img2)
#     # # Point_list.append(digit_recog(digit_image))
#     return tran(Point_list)
#     #
#     #
#     print("point_one:",Point_list)

def Point_one(img_path, code_info):
    img = cv2.imread(img_path)
    img2 = img[:, img.shape[1] // 4:img.shape[1] // 4 * 3]
    Point_info = handcart.Handcart(img2, code_info)
    Point_list = Point_info.split_and_recog()
    return Point_list


