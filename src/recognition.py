#!/usr/bin/env python
# coding: utf-8

import cv2
import numpy as np
# from paddleocr import PaddleOCR, draw_ocr
import re
# import numpy as np
# import time
# import pytesseract


def lcd_recogn(img_path):
    ocr = PaddleOCR(use_angle_cls=True, use_gpu=False)
    result = ocr.ocr(img_path, cls=True)
    txts = [line[1][0] for line in result]
    # print(txts)
    for i in range(len(txts)):
        txts[i] = re.sub('[\u4e00-\u9fa5]', '', txts[i])
    txts[2] = re.sub('[：]', '', txts[2])
    txts[2] = str(txts[2]).replace('U', 'V')
    txts[3] = re.sub('[：]','',txts[3])
    txts[3] = str(txts[3]).replace('20','Z')
    # print(s)

    return txts

def serial_recogn(img_path):
    ocr = PaddleOCR(use_angle_cls=True, use_gpu=False)
    result = ocr.ocr(img_path, cls=True)
    txts = [line[1][0] for line in result]

    return txts


def numDetect(img, loc):
    x, y, w, h = loc

    if h / w > 3:
        return 1
    else:
        a, b, c, d, e, f, g = 0, 0, 0, 0, 0, 0, 0

        # 穿针法
        line1 = img[y:y + h, x + w // 2]
        line2 = img[y + h // 4, x:x + w]
        line3 = img[y + (h // 4) * 3, x:x + w]
        # 检测竖线，从而识别a,g,d笔画
        a, b, c, d, e, f, g = 0, 0, 0, 0, 0, 0, 0
        for i in range(h):
            if line1[i] == 255:
                if i < (h // 3):
                    a = 1
                if i > 2 * (h // 3):
                    d = 1
                if i > (h // 3) and i < 2 * (h // 3):
                    g = 1
        # 检测横线line2、line3，从而识别b,f笔画并减少时间消耗
        for i in range(w):
            if line2[i] == 255:
                if i < (w // 2):
                    b = 1
                if i > (w // 2):
                    f = 1
            if line3[i] == 255:
                if i < (w // 2):
                    c = 1
                if i > (w // 2):
                    e = 1

        # 不写的眼花缭乱了，直接写就可以了
        if a and b and c and d and e and f and g == 0:
            return 0
        if a and b == 0 and c and d and e == 0 and f and g:
            return 2
        if a and b == 0 and c == 0 and d and e and f and g:
            return 3
        if a == 0 and b and c == 0 and d == 0 and e and f and g:
            return 4
        if a and b and c == 0 and d and e and f == 0 and g:
            return 5
        if a and b and c and d and e and f == 0 and g:
            return 6
        if a and b == 0 and c == 0 and d == 0 and e and f and g == 0:
            return 7
        if a and b and c and d and e and f and g:
            return 8
        if a and b and c == 0 and d and e and f and g:
            return 9

        return -1


# 斜体数字的小数点识别，在数字区域内进行boundRect识别
def detectCommaItalic(img, bounds):
    for i in range(len(bounds)):
        # print(i)
        x, y, w, h = bounds[i]
        roi = img[y + 3 * h // 4: y + h, x + 2 * w // 3: x + w]
        # cv.imshow('roi',roi)
        contours0, hierarchy0 = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # print(np.shape(hierarchy0)[1])
        # print(hierarchy0)

        boundRect0 = []
        for c in contours0:
            x, y, w, h = cv2.boundingRect(c)
            roi = cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 0, 255), 1)
            # cv.imshow('point', roi)
            # 一个筛选，可能需要看识别条件而定，有待优化——比如增加小数点的大小判断
            if h / w > 0.8 and h / w < 1.2:
                boundRect0.append([x, y, w, h])
                # 画一个方形标注一下，看看圈的范围是否正确
                # red_dil = cv.rectangle(roi, (x, y), (x+w, y+h), 255, 2)
                # cv.imshow('point', red_dil)
                return i
            if np.shape(hierarchy0)[1] == 2:
                boundRect0.append([x, y, w, h])
                # red_dil = cv.rectangle(roi, (x, y), (x+w, y+h), 255, 2)
                # cv.imshow('point', red_dil)
                return i

    return -1

def digit_detect(img):

    img_red = img[:, :, 2]
    blur = cv2.medianBlur(img_red, 3)
    _, redN_bin = cv2.threshold(img_red, 0, 255, cv2.THRESH_OTSU)  # 这个要根据数码管的光线条件进行调节，后面可以通过加上一个平均值法
    # #cv2.imshow#('redN_bin.jpg', redN_bin)
    kernel = np.ones((5, 5), np.int8)
    red_dil = cv2.dilate(redN_bin, kernel, iterations=1)  # 进行腐蚀膨胀操作   这个根据需要也可以不要  这个在split中已经处理过了
    #cv2.imshow#("redN_dil.jpg", red_dil)
    contours, hierarchy = cv2.findContours(red_dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 检测数码管中数字的轮廓
    # cv2.drawContours(img,contours,-1,(0,255,0),3)        #把边缘给画出来
    # #cv2.imshow#("lunkuo",img)
    # print(hierarchy)#表示每个轮廓的索引编号
    boundRect = []
    bounds_ = []
    boundRect_tran = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        # 一个筛选，可能需要看识别条件而定，有待优化
        if h / w > 1:
            boundRect.append([x, y, w, h])
            # 画一个方形标注一下，看看圈的范围是否正确
            # red_dil = cv2.rectangle(red_dil, (x, y), (x+w, y+h), 255, 2)
            # #cv2.imshow#('bound',red_dil)
    print(boundRect)
    line_num = 1
    digit_num = 1
    l = 0
    for i in range(len(boundRect) - 1):
        if boundRect[i][1] - boundRect[i + 1][1] > boundRect[i][3]:
            line_num = line_num + 1
            digit_num= digit_num + 1
        else:
            digit_num= digit_num + 1
    print(digit_num)
    print(line_num)
    boundRect.sort(key=lambda x: x[0], reverse=False)

    bounds = []
    bounds.append(boundRect)
    print("bounds:",bounds)

    j = 0
    i = 0
    num = 0
    num_list = []
    string = []
    string_list = []
    for j in range(line_num):
        for i in range(digit_num // 1):
            num = num * 10 + numDetect(red_dil, bounds[j][i])
            string += str(numDetect(red_dil, bounds[j][i]))
            string = "".join(string)
            # print(string)
        num_list.append(num)
        string_list.append(string)
        string = []
        num = 0

    # string_list = "".join(string)
    # print('numDetect:', num_list)
    # print('string_list:', string_list)
    # print(type(string_list))

    j = 0
    Point = []
    string_mid = []
    for j in range(line_num):
        Point.append(detectCommaItalic(red_dil, bounds[j]))
        string_mid = list(string_list[j])
        string_mid.insert(Point[j] + 1, '.')
        string_list[j] = "".join(string_mid)
        # string.insert(Point[j]+1,'.')
        # string_list[j] = "".join(string)

    j = 0
    for j in range(line_num):
        # print(len(bounds[j]))
        num_list[j] = num_list[j] / pow(10, len(bounds[j]) - Point[j] - 1)

    # print("num_list:", num_list)
    # print("string_list:", string_list)
    return string_list[0]   #这个后面有待完善，这里应该是把string转换成字符串形式
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()



