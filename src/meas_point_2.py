import copy

import cv2
import numpy as np
import os
import recognition
import split


# def #cvshow1000#(name, img):
#      #cv2.namedWindow#(name, cv2.WINDOW_NORMAL)
#      #cv2.resizeWindow#(name, 1000, 750)
#      ##cv2.imshow##(name, img)

def gain_h_and_w(img):
    h = img.shape[0]
    w = img.shape[1]
    return h,w


def dispatchNum_split(img):
    image = img
    split_h = image.shape[0]
    split_w = image.shape[1]
    image_split = image[split_h // 4:split_h // 4 * 3, split_w // 4:3 * split_w // 4]
    # ##cv2.imshow##("image_split",image_split)
    gray = cv2.cvtColor(image_split, cv2.COLOR_BGR2GRAY)
    # #cv2.namedWindow#("gray",cv2.WINDOW_NORMAL)
    # #cv2.resizeWindow#("gray",1000,750)
    # ##cv2.imshow##("gray",gray)

    image_g = cv2.GaussianBlur(gray, (3, 3), 0)
    # #cv2.namedWindow#("gauss", 0)
    # #cv2.resizeWindow#("gauss", 1000, 750)
    # ##cv2.imshow##("gauss", image_g)

    ret, image_er = cv2.threshold(image_g, 0, 255, cv2.THRESH_OTSU)  # OTSU最大类间方差法
    # #cv2.namedWindow#("er", cv2.WINDOW_NORMAL)
    # #cv2.resizeWindow#("er", 1000, 750)
    # ##cv2.imshow##("er", image_er)

    contours, hierarchy = cv2.findContours(image_er, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 找轮廓
    # cv2.drawContours(image_split, contours, -1, (0, 255, 0), 3)  # 把边缘给画出来
    # #cv2.namedWindow#("luokuo",0)
    # #cv2.resizeWindow#("luokuo",1000,750)
    # ##cv2.imshow##("luokuo",image_split)
    boundRect = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        # 一个筛选，可能需要看识别条件而定，有待优化
        if w / h > 1.5 and w < 0.9 * split_w and w > 0.1 * split_w and h < 0.9 * split_h and h > 0.1 * split_h:
            boundRect.append([x, y, w, h])
            # 画一个方形标注一下，看看圈的范围是否正确
            # red_dil = cv2.rectangle(image_split, (x, y), (x + w, y + h), 255, 2)
            # print(x, y, w, h)
            # #cv2.namedWindow#("bound", 0)
            # #cv2.resizeWindow#("bound", 1000, 750)
            # ##cv2.imshow##('bound', red_dil)
    # print(boundRect)
    # print(np.shape(boundRect))
    # 暂时通过最大值来判断
    a = np.array(boundRect)
    maxindex = a.argmax(axis=0)
    # print(maxindex)
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
    # ##cv2.imshow##("image_out",image_out)
    print(dispatchNum_coordinate)
    diaoduhao40 = cv2.rectangle(image_split, (dispatchNum_coordinate_x, dispatchNum_coordinate_y), (dispatchNum_coordinate_x + dispatchNum_coordinate_w,
                                                                                                dispatchNum_coordinate_y + dispatchNum_coordinate_h), (0,0,255), 2)
    # #cv2.namedWindow#("40",0)
    # #cv2.resizeWindow#("40",1000,750)
    ##cv2.imshow##("40",diaoduhao40)
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
    # ##cv2.imshow##("gray", gray)
    ret, image_er = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)  # OTSU最大类间方差法
    # #cv2.namedWindow#("er", cv2.WINDOW_NORMAL)
    # #cv2.resizeWindow#("er", 1000, 750)
    ##cv2.imshow##("er", image_er)
    contours, hierarchy = cv2.findContours(image_er, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 找轮廓
    # cv2.drawContours(image_split, contours, -1, (0, 255, 0), 3)  # 把边缘给画出来
    # #cv2.namedWindow#("luokuo",0)
    # #cv2.resizeWindow#("luokuo",1000,750)
    # ##cv2.imshow##("luokuo",image_split)
    boundRect = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        # 一个筛选，可能需要看识别条件而定，有待优化
        if w / h > 1 and w < 0.8 * split_w and w > 0.1 * split_w and h < 0.8 * split_h and h > 0.1 * split_h:
            boundRect.append([x, y, w, h])
            # 画一个方形标注一下，看看圈的范围是否正确
            # red_dil = cv2.rectangle(image_split, (x, y), (x + w, y + h), 255, 2)
            # # print(x, y, w, h)
            # #cv2.namedWindow#("bound", 0)
            # #cv2.resizeWindow#("bound", 1000, 750)
            # ##cv2.imshow##('bound', red_dil)
    # print(boundRect)
    # print(np.shape(boundRect))
    # 暂时通过最大值来判断
    a = np.array(boundRect)
    maxindex = a.argmax(axis=0)
    # print(maxindex)
    led_bound = []
    led_bound = (boundRect[maxindex[2]])
    # print(led_bound)
    led_coordinate_x = led_bound[0]
    led_coordinate_y = led_bound[1]
    led_coordinate_w = led_bound[2]
    led_coordinate_h = led_bound[3] // 2
    led_coordinate = np.array([led_coordinate_x, led_coordinate_y, led_coordinate_w, led_coordinate_h])
    image_out = image_split[led_coordinate_y:led_coordinate_h + led_coordinate_y,
                            led_coordinate_x: led_coordinate_w + led_coordinate_x]
    return image_out


#这个是识别函数，主要将分割出来的L1、2、3的亮灭状态给区分出来，1表示亮，0表示灭
def L_led_recog(image):
    # ##cv2.imshow##("image", image)
    # print("image_w:",image.shape[1],"image_h:",image.shape[0])
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)
    # ##cv2.imshow##("gray", gray)
    ret, image_er = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)  # OTSU最大类间方差法
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # image_er = cv2.morphologyEx(image_er, cv2.MORPH_OPEN, kernel,iterations=3)
    # ##cv2.imshow##("image_er",image_er)
    # 这里 灭的灯识别不出来
    # print(image_er.shape[0])
    canny = cv2.Canny(image_er, 50, 150)
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
            # ##cv2.imshow##("image_yuan", image_yuan)
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
            if circles[0][i][0] <= image.shape[1] // 5 * 2 and average_v > 100:
                led_list[0] = '亮'

            if circles[0][i][0] > image.shape[1] // 5 * 2 and circles[0][i][0] < image.shape[1] // 5 * 3 and average_v > 100:
                led_list[1] = '亮'

            if circles[0][i][0] >= image.shape[1] // 5 * 3 and average_v > 100:
                led_list[2] = '亮'

        print("L_led:",led_list)
        ##cv2.imshow##("yuan", image)
        return led_list

#将远方开关给分割出来
def APTkey_split(img):
    image = img
    image_h ,image_w = gain_h_and_w(image)
    image_split = image[image.shape[0]//3*2:image.shape[0], 0:image.shape[1]]
    # ##cv2.imshow##("image_split",image_split)
    gray = cv2.cvtColor(image_split, cv2.COLOR_BGR2GRAY)
    # ##cv2.imshow##("gray", gray)
    ret, image_er = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)  # OTSU最大类间方差法
    contours, hierarchy = cv2.findContours(image_er, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 找轮廓
    # cv2.drawContours(image_split, contours, -1, (0, 255, 0), 3)  # 把边缘给画出来
    # ##cv2.imshow##("contour",image_split)
    split_h, split_w = gain_h_and_w(image_split)
    boundRect = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        # 一个筛选，可能需要看识别条件而定，有待优化
        if w / h < 1.2 and w/h >0.8 and w < 0.8 * split_w and h < 0.8 * split_h and h > 0.1 * split_h:
            boundRect.append([x, y, w, h])
            # 画一个方形标注一下，看看圈的范围是否正确
            # red_dil = cv2.rectangle(image_split, (x, y), (x + w, y + h), 255, 2)
            # print(x, y, w, h)
            # ##cv2.imshow##('bound', red_dil)
    a = np.array(boundRect)
    maxindex = a.argmax(axis=0)
    b = np.delete(a,maxindex[2],axis=0)
    maxindex_second = b.argmax(axis=0)
    # print(b[maxindex_second[2]])
    if maxindex[2]<=maxindex_second[2]:
        maxindex_second = [i+1 for i in maxindex_second]
    #比较两个开关的位置
    if boundRect[maxindex[2]][0] <= boundRect[maxindex_second[2]][0]:
        image_return_yuanfang = image_split[boundRect[maxindex[2]][1]:boundRect[maxindex[2]][1]+boundRect[maxindex[2]][3],
                           boundRect[maxindex[2]][0]:boundRect[maxindex[2]][0]+boundRect[maxindex[2]][2]]
        image_return_yuhe = image_split[boundRect[maxindex_second[2]][1]:boundRect[maxindex_second[2]][1]+boundRect[maxindex_second[2]][3],
                           boundRect[maxindex_second[2]][0]:boundRect[maxindex_second[2]][0]+boundRect[maxindex_second[2]][2]]
        return image_return_yuanfang, image_return_yuhe
    else:
        image_return_yuanfang = image_split[boundRect[maxindex_second[2]][1]:boundRect[maxindex_second[2]][1]+boundRect[maxindex_second[2]][3],
                           boundRect[maxindex_second[2]][0]:boundRect[maxindex_second[2]][0]+boundRect[maxindex_second[2]][2]]
        image_return_yuhe = image_split[boundRect[maxindex[2]][1]:boundRect[maxindex[2]][1]+boundRect[maxindex[2]][3],
                           boundRect[maxindex[2]][0]:boundRect[maxindex[2]][0]+boundRect[maxindex[2]][2]]
        return image_return_yuanfang, image_return_yuhe

def APTkey_yuanfang_recogn(img):

    h,w = gain_h_and_w(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    ret, image_er = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)  # OTSU最大类间方差法
    ##cv2.imshow##("img_er",image_er)
    img_down = image_er[h//4*3:h, 0:w]
    # ##cv2.imshow##("down",img_down)
    img_right = image_er[0:h, w//4*3:w]
    # ##cv2.imshow##("right",img_right)
    len_down1 = len(img_down[img_down==0])
    # len_down2 = len(img_down[img_down==255])
    len_right1 = len(img_right[img_right==0])
    # len_right2 = len(img_right[img_right==255])
    #0表示开关为竖着的状态（远地开关），1表示横着的状态（就地开关）
    if len_down1>len_right1:
        return '远方'
    else:
        return '就地'

def APTkey_yuhe_recogn(img):

    h,w = gain_h_and_w(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    ret, image_er = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)  # OTSU最大类间方差法
    ##cv2.imshow##("img_er",image_er)
    img_down = image_er[h//4*3:h, 0:w]
    # ##cv2.imshow##("down",img_down)
    img_right = image_er[0:h, w//4*3:w]
    # ##cv2.imshow##("right",img_right)
    len_down1 = len(img_down[img_down==0])
    # len_down2 = len(img_down[img_down==255])
    len_right1 = len(img_right[img_right==0])
    # len_right2 = len(img_right[img_right==255])
    #0表示开关为竖着的状态（远地开关），1表示横着的状态（就地开关）
    if len_down1>len_right1:
        return '预合合后'
    else:
        return '预合分后'



#通过这个函数可以直接获得开关的开关状态，1表示打向右边（开），0表示打向左边（关）
def lighting_key_split(image_path):
    image = cv2.imread(image_path)
    split_h = image.shape[0] // 2
    split_w = image.shape[1] // 2
    # print("split_h:", split_h)
    # print("split_w:", split_w)
    image_split = image[split_h: split_h * 2, split_w: split_w*2]
    gray = cv2.cvtColor(image_split, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)
    # gray = cv2.GaussianBlur(gray,(3,3),0)
    # ##cv2.imshow##("gray", gray)
    ret, image_er = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)  # OTSU最大类间方差法
    # #cv2.namedWindow#("er", cv2.WINDOW_NORMAL)
    # #cv2.resizeWindow#("er", 1000, 750)
    ##cv2.imshow##("er", image_er)
    contours, hierarchy = cv2.findContours(image_er, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 找轮廓
    # cv2.drawContours(image_split, contours, -1, (0, 255, 0), 3)  # 把边缘给画出来
    # #cv2.namedWindow#("luokuo",0)
    # #cv2.resizeWindow#("luokuo",1000,750)
    # ##cv2.imshow##("luokuo",image_split)
    circles = cv2.HoughCircles(image_er, cv2.HOUGH_GRADIENT, 1, 300, param1=300, param2=50, minRadius=0,
                               maxRadius=image_er.shape[0] // 2)                                             #不同测点对于圆的判定阈值会不同，可以尝试 自适应阈值，把阴影个去掉，还有就是对于圆检测再做尝试

    # print(len(circles))
    # print("shape[1]:",image_split.shape[1]//10)
    circles = np.array([])
    if circles is not None:
        circles = np.uint16(np.around(circles))  # around对数据四舍五入，为整数
        leng = len(circles[0][0])
        for i in range(leng):
            if circles[0][0][i][0] < image_split.shape[0]//8:
                # print("i:",circles[0][0][i][0])
                circles = np.delete(circles[0][0],i,0)

        for i in circles[0, :]:
            cv2.circle(image_split, (i[0], i[1]), i[2], (0, 0, 255), 2)  # 画圆
            cv2.circle(image_split, (i[0], i[1]), 2, (0, 0, 255), 2)  # 画圆心
    print(circles)
    print(circles)
    # #cv2.namedWindow#("yuan",0)
    # #cv2.resizeWindow#("yuan",1000,750)
    ##cv2.imshow##("yuan", image_split)
    image_zhong = image_split[circles[0][0][1]-circles[0][0][2]:circles[0][0][1]+circles[0][0][2], circles[0][0][0]-circles[0][0][2]:circles[0][0][0]+circles[0][0][2]]
    image_yuan = gray[circles[0][0][1]-circles[0][0][2]:circles[0][0][1]+circles[0][0][2], circles[0][0][0]-circles[0][0][2]:circles[0][0][0]+circles[0][0][2]]
    ##cv2.imshow##("image_yuan",image_yuan)
    edge = cv2.Canny(image_yuan,50,150)
    ##cv2.imshow##("bound", edge)
    lines = cv2.HoughLinesP(edge, 1, np.pi / 180, threshold=30,minLineLength=30)
    for x1, y1, x2, y2 in lines[:, 0]:
        cv2.line(image_zhong, (x1, y1), (x2, y2), (0, 0, 255), 2)
    # print("line",lines)
    ##cv2.imshow##("xian,",image_zhong)
    k = []
    for x1, y1, x2, y2 in lines[:, 0]:
        k.append((y1-y2)/(x1-x2))
    k_sum = sum(k)
    if k_sum < -1/5:
        return 1
    else:
        return 0



def lianpian_split(img):
    image = img
    image_h, image_w = gain_h_and_w(image)
    # print(image_h,image_w)
    image_split = image[image_h // 2 : image_h, image_w//2:image_w]
    # #cvshow1000#("image_split",image_split)
    gray = cv2.cvtColor(image_split, cv2.COLOR_BGR2GRAY)
    # ##cv2.imshow##("gray", gray)
    ret, image_er = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)  # OTSU最大类间方差法
    contours, hierarchy = cv2.findContours(image_er, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 找轮廓
    # cv2.drawContours(image_split, contours, -1, (0, 255, 0), 3)  # 把边缘给画出来
    # #cvshow1000#("contour",image_er)
    # #cvshow1000#("luokuo",image_split)
    # canny = cv2.Canny(gray, 150, 300)
    # #cvshow1000#("canny",canny)
    split_h, split_w = gain_h_and_w(image_split)
    boundRect = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        # 一个筛选，可能需要看识别条件而定，有待优化
        if w / h < 5 and w / h > 1 and w < 0.8 * split_w and h < 0.8 * split_h and h > 0.1 * split_h:
            boundRect.append([x, y, w, h])
            # 画一个方形标注一下，看看圈的范围是否正确
            # red_dil = cv2.rectangle(image_split, (x, y), (x + w, y + h), 255, 2)
            # #cvshow1000#("red",red_dil)
    # print(boundRect)
    maxindex = np.argmax(boundRect,axis=0)
    black_area = boundRect[maxindex[2]]
    lianpian_area = [black_area[0]-black_area[2]//2,black_area[1]-black_area[3]*1.55,black_area[2]//2*5,black_area[3]*1.25]
    lianpian_area = [int (i) for i in lianpian_area]
    # print("lianpian",lianpian_area)
    lianpian_image = image_split[lianpian_area[1]:lianpian_area[1]+lianpian_area[3],lianpian_area[0]:lianpian_area[0]+lianpian_area[2]]
    ##cv2.imshow##("img",lianpian_image)
    # black_img = image_split[black_area[1]:black_area[1]+black_area[3],black_area[0]:black_area[0]+black_area[2]]
    # ##cv2.imshow##("bla",black_img)

    return lianpian_image, lianpian_area

def mark_recog(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    l_yellow = np.array([[11, 43, 46]])
    h_yellow = np.array([34, 255, 255])
    mask_yellow = cv2.inRange(hsv, l_yellow, h_yellow)
    res = cv2.bitwise_and(img, img, mask=mask_yellow)
    l_red1 = np.array([156, 100, 0])  # 提取颜色的低值 red [156, 43, 46]
    h_red1 = np.array([180, 255, 120])  # 提取颜色的高值 [180, 255, 255]
    l_red2 = np.array([0, 100, 0])  # 提取颜色的低值 red [0, 43, 46]
    h_red2 = np.array([10, 255, 120])  # 提取颜色的高值 [10, 255, 255]
    mask_red1 = cv2.inRange(hsv, lowerb=l_red1, upperb=h_red1)
    mask_red2 = cv2.inRange(hsv, lowerb=l_red2, upperb=h_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    mask = cv2.bitwise_or(mask_red, mask_yellow)
    ##cv2.imshow##("mask_yellow", mask_yellow)
    ##cv2.imshow##("mask_red", mask_red)
    ##cv2.imshow##("mask",mask)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 找轮廓
    # cv2.drawContours(img, contours, -1, (0, 255, 0), 3)  # 把边缘给画出来
    # ##cv2.imshow##("bound",img)
    boundRect = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        # 一个筛选，可能需要看识别条件而定，有待优化
        if  h < img.shape[0] * 0.95 and w / h <2 and w/h>0.1 and h >img.shape[0]//6:
            boundRect.append([x, y, w, h])
            # 画一个方形标注一下，看看圈的范围是否正确
            red_dil = cv2.rectangle(img, (x, y), (x + w, y + h), 255, 2)
            ##cv2.imshow##("red",red_dil)
    print(boundRect)

    if boundRect is not None:
        boundRect.sort(key=lambda x: x[0], reverse=False)
        lianpian_list = []
        for i in range(len(boundRect)):
            if boundRect[i][2] < boundRect[i][3]*0.8:
                lianpian_list.insert(i, '合')  # 0代表闭合，1代表打开
            else:
                lianpian_list.insert(i, '分')
        print("lianpian:",lianpian_list)
        return lianpian_list
    else:
        lianpian_list = [' ', ' ']
        return lianpian_list





def lianpian_recog(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    l_yellow = np.array([[11, 43, 46]])
    h_yellow = np.array([34, 255, 255])
    mask_yellow = cv2.inRange(hsv, l_yellow, h_yellow)
    res = cv2.bitwise_and(img, img, mask=mask_yellow)
    l_red1 = np.array([156, 100, 0])  # 提取颜色的低值 red [156, 43, 46]
    h_red1 = np.array([180, 255, 120])  # 提取颜色的高值 [180, 255, 255]
    l_red2 = np.array([0, 100, 0])  # 提取颜色的低值 red [0, 43, 46]
    h_red2 = np.array([10, 255, 120])  # 提取颜色的高值 [10, 255, 255]
    mask_red1 = cv2.inRange(hsv, lowerb=l_red1, upperb=h_red1)
    mask_red2 = cv2.inRange(hsv, lowerb=l_red2, upperb=h_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    mask = cv2.bitwise_or(mask_red, mask_yellow)
    ##cv2.imshow##("mask_yellow", mask_yellow)
    ##cv2.imshow##("mask_red", mask_red)
    ##cv2.imshow##("mask",mask)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 找轮廓
    # cv2.drawContours(img, contours, -1, (0, 255, 0), 3)  # 把边缘给画出来
    # ##cv2.imshow##("bound",img)
    boundRect = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        # 一个筛选，可能需要看识别条件而定，有待优化
        if  h < img.shape[0] * 0.95 and w / h <2 and w/h>0.1 and h >img.shape[0]//3:
            boundRect.append([x, y, w, h])
            # 画一个方形标注一下，看看圈的范围是否正确
            red_dil = cv2.rectangle(img, (x, y), (x + w, y + h), 255, 2)
            ##cv2.imshow##("red",red_dil)
    print(boundRect)

    if boundRect is not None:
        boundRect.sort(key=lambda x: x[0], reverse=False)
        lianpian_list = []
        for i in range(len(boundRect)):
            if boundRect[i][2] < boundRect[i][3]*0.8:
                lianpian_list.insert(i, '合')  # 0代表闭合，1代表打开
            else:
                lianpian_list.insert(i, '分')
        print("lianpian:",lianpian_list)
        return lianpian_list
    else:
        lianpian_list = [' ', ' ']
        return lianpian_list

#通过这个函数可以直接获得测点2 开关的开关状态，1表示打向右边（开），0表示打向左边（关）
def b_w_key_split2(image, lianpian_location):
    img = image
    h,w = gain_h_and_w(img)
    b_w_key_area = [w//2+lianpian_location[0],h//2+lianpian_location[1]-lianpian_location[3],lianpian_location[2],lianpian_location[3]]
    print(b_w_key_area)
    b_w_key_img = img[b_w_key_area[1]:b_w_key_area[1]+b_w_key_area[3],b_w_key_area[0]:b_w_key_area[0]+b_w_key_area[2]]
    ##cv2.imshow##("split",b_w_key_img)
    return b_w_key_img, b_w_key_area

def b_w_key_recog2(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)
    # gray = cv2.GaussianBlur(gray,(3,3),0)
    # ##cv2.imshow##("gray", gray)
    ret, image_er = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)  # OTSU最大类间方差法
    circles = cv2.HoughCircles(image_er, cv2.HOUGH_GRADIENT, 1, 200, param1=200, param2=30, minRadius=20,
                               maxRadius=image_er.shape[0] //2)
    # circles = np.array([])
    if circles is not None:
        circles = np.uint16(np.around(circles))  # around对数据四舍五入，为整数
        # leng = len(circles[0][0])
        # for i in range(leng):
        #     if circles[0][0][i][0] < image_split.shape[0] // 8:
        #         # print("i:",circles[0][0][i][0])
        #         circles = np.delete(circles[0][0], i, 0)

        for i in circles[0, :]:
            cv2.circle(img, (i[0], i[1]), i[2], (0, 0, 255), 2)  # 画圆
            cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 2)  # 画圆心
        # print(circles)
        cir = []
        for i in range(len(circles[0])):
            cir.append(circles[0][i])
        # print(cir[0])
        cir.sort(key=lambda x: x[0], reverse=False)
        # print(len(cir))
        ##cv2.imshow##("yuan", img)
        k = []
        k_sum = []
        key_list = []
        for i in range(len(cir)):
            image_zhong = img[cir[i][1] - cir[i][2]:cir[i][1] + cir[i][2],
                          cir[i][0] - cir[i][2]:cir[i][0] + cir[i][2]]
            image_yuan = gray[cir[i][1] - cir[i][2]:cir[i][1] + cir[i][2],
                         cir[i][0] - cir[i][2]:cir[i][0] + circles[0][i][2]]
            # ##cv2.imshow##("image_yuan", image_yuan)
            edge = cv2.Canny(image_yuan, 50, 150)
            ##cv2.imshow##("bound", edge)
            lines = cv2.HoughLinesP(edge, 1, np.pi / 180, threshold=30, minLineLength=30)
            for x1, y1, x2, y2 in lines[:, 0]:
                cv2.line(image_zhong, (x1, y1), (x2, y2), (0, 0, 255), 2)
            print("line", lines)
            ##cv2.imshow##("xian,", image_zhong)
            for x1, y1, x2, y2 in lines[:, 0]:
                k.append((y1 - y2) / (x1 - x2))
            k_sum.append(sum(k))
            k = []
        print(k_sum)
        for i in range(len(k_sum)):
            if k_sum[i] < -1 / 5:
                key_list.insert(i, '开')
            else:
                key_list.insert(i, '关')
        print("key:",key_list)
        return key_list
    else:
        key_list = [99, 99]
        return key_list

        # if k_sum < -1 / 5:
        #     return 1
        # else:
        #     return 0

def instruct_led_split(image, bwkey_area):
    img = image
    h,w = gain_h_and_w(img)
    led_location =[bwkey_area[0],bwkey_area[1]-bwkey_area[3],bwkey_area[2],bwkey_area[3]]
    # print("led",led_location)
    led_img = img[led_location[1]:led_location[1]+led_location[3],led_location[0]:led_location[0]+led_location[2]]
    ##cv2.imshow##("led_img",led_img)
    return led_img, led_location

def instruct_led_recogn(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)
    # gray = cv2.GaussianBlur(gray,(3,3),0)
    # ##cv2.imshow##("gray", gray)
    canny = cv2.Canny(gray, 50, 150)
    ##cv2.imshow##("canny",canny)
    ret, image_er = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)  # OTSU最大类间方差法
    ##cv2.imshow##("led_er",image_er)
    circles = cv2.HoughCircles(canny, cv2.HOUGH_GRADIENT, 1, 200, param1=150, param2=20, minRadius=50,
                               maxRadius=image_er.shape[0] //4)
    # circles = np.array([])
    if circles is not None:
        circles = np.uint16(np.around(circles))  # around对数据四舍五入，为整数
        # leng = len(circles[0][0])
        # for i in range(leng):
        #     if circles[0][0][i][0] < image_split.shape[0] // 8:
        #         # print("i:",circles[0][0][i][0])
        #         circles = np.delete(circles[0][0], i, 0)

        for i in circles[0, :]:
            cv2.circle(img, (i[0], i[1]), i[2], (255,0,0), 2)  # 画圆
            cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 2)  # 画圆心
        print(circles)
        ##cv2.imshow##("sangeyuan", img)

        cir = []
        for i in range(len(circles[0])):
            cir.append(circles[0][i])
        print(cir[0])
        cir.sort(key=lambda x: x[0], reverse=False)
        print(len(cir))
        ##cv2.imshow##("yuan", img)
        led_list = []
        for i in range(len(cir)):
            image_zhong = img[cir[i][1] - cir[i][2]:cir[i][1] + cir[i][2],
                          cir[i][0] - cir[i][2]:cir[i][0] + cir[i][2]]
            # ##cv2.imshow##("image_yuan", image_yuan)
            hsv = cv2.cvtColor(image_zhong, cv2.COLOR_RGB2HSV)
            H, S, V = cv2.split(hsv)
            # print("HSV", H, S, V)
            v = V.ravel()[np.flatnonzero(V)]  # 亮度非零的值
            average_v = sum(v) / len(v)
            print("average_v", average_v)
            if average_v < 180:
                led_list.insert(i, '灭')
            else:
                led_list.insert(i, '亮')
        print("led_list", led_list)
        return led_list

def handcar_split(image, last_location):
    img = image
    h,w = gain_h_and_w(img)
    handcar_location =[last_location[0],last_location[1]-last_location[3]//4*3, last_location[2], last_location[3]]
    # print("handcar_",handcar_location)
    led_img = img[handcar_location[1]:handcar_location[1]+handcar_location[3],handcar_location[0]:handcar_location[0]+handcar_location[2]]
    ##cv2.imshow##("hand_img",led_img)
    return led_img, handcar_location

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
    ##cv2.imshow##("led_er",image_er)
    bin = cv2.Canny(image_er, 50, 150, apertureSize=3)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # 矩形结构
    bin = cv2.morphologyEx(bin, cv2.MORPH_CLOSE, kernel)
    ##cv2.imshow##("bin", bin)
    contours, _hierarchy = cv2.findContours(bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, (0, 255, 0), 3)  # 把边缘给画出来
    ##cv2.imshow##("bin2", img)
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
    # #cv2.namedWindow#("bound", 0)
    # #cv2.resizeWindow#("bound", 1000, 750)
    ##cv2.imshow##('bound', img)

    # x_min = np.min(squares[0],  axis = 0)[0]
    # y_min = np.min(squares[0],  axis = 0)[1]
    # x_max = np.max(squares[0],  axis = 0)[0]
    # y_max = np.max(squares[0],  axis = 0)[1]
    # print(x_min,x_max,y_min,y_max )
    # img_return = img[y_min:y_max,x_min:x_max]
    # ##cv2.imshow##("return", img_return)
    #
    # return img_return

def handcar_led_recog(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    l_yellow = np.array([[15, 43, 46]])
    h_yellow = np.array([34, 255, 255])
    mask_yellow = cv2.inRange(hsv, l_yellow, h_yellow)
    res = cv2.bitwise_and(img, img, mask=mask_yellow)
    ##cv2.imshow##("mask_yellow", mask_yellow)

    contours, hierarchy = cv2.findContours(mask_yellow, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 找轮廓
    boundRect = []
    color = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        # 一个筛选，可能需要看识别条件而定，有待优化
        if  h < img.shape[0] * 0.95 and w/h >3 and w>img.shape[1]//8:
            boundRect.append([x, y, w, h])
            # 画一个方形标注一下，看看圈的范围是否正确
            red_dil = cv2.rectangle(img, (x, y), (x + w, y + h), 255, 2)
            ##cv2.imshow##("handcar_red",red_dil)
    if boundRect is not None and len(boundRect) == 3:
        boundRect.sort(key=lambda x: x[0], reverse=False)
        print(boundRect)
        for i in range(len(boundRect)):
            img_area = [boundRect[i][0] + boundRect[i][3]//3*2, boundRect[i][1] - boundRect[i][3]//2*7, boundRect[i][2]-boundRect[i][3], boundRect[i][3]//2*5]
            handcar_img = img[img_area[1]:img_area[1]+img_area[3], img_area[0]:img_area[0]+img_area[2]]
            # ##cv2.imshow##("obj{}".format(i), handcar_img)
            hsv = cv2.cvtColor(handcar_img, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
            object_H = np.argmax(hist)
            print("object_H:", object_H)

            if object_H >= 156 and object_H <= 180:
                color.append("合闸")
            elif object_H >= 0 and object_H <= 10:
                color.append("合闸")
            elif object_H > 10 and object_H <= 34:
                color.append("分闸")
            else:
                color.append(" ")
            print("color:", color)
        return color
    else:
        color = [' ', ' ', ' ']
        return color


        # index = []
        # w_average = np.mean(boundRect, axis=0)[2]
        # print(w_average)
        # for i in range(len(boundRect)):
        #     if len(boundRect) == 3:
        #         boundRect_.append(boundRect[i])
        #     if boundRect[i][2] < w_average:
        #         boundRect_.append(boundRect[i])
        # boundRect_mid = copy.deepcopy(boundRect_)
        # for i in range(len(boundRect_mid) - 1):
        #     if boundRect_mid[i+1][0] - boundRect_mid[i][0] < boundRect_mid[i][2]:
        #         boundRect_.pop(i)
        # color = []
        # for i in range(len(boundRect_)):
        #     img_quar = img[boundRect_[i][1]:boundRect_[i][1] + boundRect_[i][3],
        #                boundRect_[i][0]:boundRect_[i][0] + boundRect_[i][2]]
        #     hsv = cv2.cvtColor(img_quar, cv2.COLOR_BGR2HSV)
        #     # H,S,V = np.split(hsv)
        #     hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
        #     object_H = np.argmax(hist)
        #     print("object_H:", object_H)
        #
        #     if object_H >= 156 and object_H <= 180:
        #         color.append("合闸")
        #     elif object_H >= 0 and object_H < 10:
        #         color.append("合闸")
        #     elif object_H > 10 and object_H <= 34:
        #         color.append("分闸")
        #     else:
        #         color.append(" ")
        #     print("color:", color)
        # return color


def running_led_split(image, last_location):
    img = image
    h,w = gain_h_and_w(img)
    running_location =[last_location[0] - last_location[3], last_location[1], last_location[3], last_location[3]*2]
    # print("running",running_location)
    led_img = img[running_location[1]:running_location[1]+running_location[3],running_location[0]:running_location[0]+running_location[2]]
    ##cv2.imshow##("run_led",led_img)
    return led_img, running_location

def running_led_recog(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    ##cv2.imshow##("gray", gray)
    canny = cv2.Canny(gray, 50,150)
    ##cv2.imshow##("canny",canny)
    ret, image_er = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)  # OTSU最大类间方差法
    contours, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 找轮廓
    # cv2.drawContours(img, contours, -1, (0, 255, 0), 3)  # 把边缘给画出来
    # ##cv2.imshow##("bianyuan",img)
    boundRect = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        # 一个筛选，可能需要看识别条件而定，有待优化
        if  h < img.shape[0] * 0.95 and w/h >3 and w/h<6 and w > img.shape[1]//10 :
            boundRect.append([x, y, w, h])
            # 画一个方形标注一下，看看圈的范围是否正确
            # red_dil = cv2.rectangle(img, (x, y), (x + w//2, y + h), 255, 2)
            # ##cv2.imshow##("yunixng_led",red_dil)
    led_list = []
    if boundRect is not None:
        # boundRect = np.unique(boundRect)
        boundRect = list(set([tuple(t) for t in boundRect]))
        boundRect.sort(key=lambda x: x[1], reverse=False)
        print(boundRect)
        color = []

        for i in range(len(boundRect)):
            img_quar = img[boundRect[i][1]:boundRect[i][1] + boundRect[i][3],
                       boundRect[i][0]:boundRect[i][0] + boundRect[i][2] // 2]
            hsv = cv2.cvtColor(img_quar, cv2.COLOR_BGR2HSV)
            H, S, V = cv2.split(hsv)
            # print("HSV", H, S, V)
            v = V.ravel()[np.flatnonzero(V)]  # 亮度非零的值
            average_v = sum(v) / len(v)
            print("average_v", average_v)
            if average_v < 90:
                led_list.insert(i, '灭')
            else:
                led_list.insert(i, '亮')
        if len(led_list) == 6:
            led_list.pop(3)
        print("led_list", led_list)
        return led_list
    else:
        led_list = ['灭','灭','灭','灭','灭' ]
        return led_list


def digit_recog(digit_image):
    digit_list = []
    for i in range(len(digit_image)):
        # ##cv2.imshow##('obj_digit{}'.format(i),digit_image[i])
        fx = 2
        fy = 2
        digit_large = cv2.resize(digit_image[i], (0, 0), fx=fx, fy=fy, interpolation=cv2.INTER_LINEAR)
        digit_list.append(recognition.digit_detect(digit_large))
    print(digit_list)
    return digit_list

def tran(list):
    list_str = str(list).replace("[", "").replace("]", "")
    return eval(f"[{list_str}]")

def Point_two(file_name2, code_info):
    # img1 = cv2.imread(file_name1)
    img2 = cv2.imread(file_name2)
    Point_list = []
    # num_img = dispatchNum_split(img1)
    # # ##cv2.imshow##("num", num_img)
    # Point_list.append(dispatchNum_recog(num_img))
    L_led_img = L_led_split(img2)
    Point_list.append(L_led_recog(L_led_img))
    yuanfang_img, yuhe_img = APTkey_split(img2)
    ##cv2.imshow##("yuanfnag",yuanfang_img)
    Point_list.append(APTkey_yuanfang_recogn(yuanfang_img))
    Point_list.append(APTkey_yuhe_recogn(yuhe_img))
    lianpian_img, lianpian_area = lianpian_split(img2)
    Point_list.append(lianpian_recog(lianpian_img))
    b_w_key_img, b_w_key_area = b_w_key_split2(img2, lianpian_area)
    Point_list.append(b_w_key_recog2(b_w_key_img))
    instruct_led_img, instruct_led_area = instruct_led_split(img2, b_w_key_area)
    Point_list.append(instruct_led_recogn(instruct_led_img))
    handcar_img, handcar_area = handcar_split(img2, instruct_led_area)
    Point_list.append(handcar_led_recog(handcar_img))
    running_led_img, running_led_area = running_led_split(img2, handcar_area)
    Point_list.append(running_led_recog(running_led_img))
    # digit_img = split.detect_and_split_1(img2)
    # Point_list.append(digit_recog(digit_img))
    return tran(Point_list)


    print("point_two:",Point_list)


#
# if __name__ == '__main__':
#     file_name = 'E:\\desktop\\images2\\2-2.JPG'
#     _, lianpian_area = lianpian_split(file_name)
#     # list = lianpian_recog(img)
#     img, key_area = b_w_key_split2(file_name,lianpian_area)
#     # key = b_w_key_recogn2(img)
#     _, led_area = instruct_led_split(file_name,  key_area)
#     # instruct_led_recogn(img)
#     _, handcar_led_area = handcar_split(file_name, led_area)
#     # handcar_led_recog(img)
#     img = yunxing_led_split(file_name, handcar_led_area)
#     yunxing_led_recog(img)
#
#     # print(key)
#     # ##cv2.imshow##("img",img)
#     # APTkey_recogn(img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()