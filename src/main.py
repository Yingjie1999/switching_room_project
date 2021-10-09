# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

# import pytesseract
import os
import cv2
import numpy as np
from mycode import meas_point_1
# import recognition
import split
import matplotlib.pyplot as plt
import pandas as pd
import ultis
# import xlsxwriter

from PIL import Image

def OCR_digit(image_path):
    # 导入OCR安装路径，如果设置了系统环境，就可以不用设置了
    # pytesseract.pytesseract.tesseract_cmd = r"D:\Program Files\Tesseract-OCR\tesseract.exe"
    # 打开要识别的图片

    src = cv2.imread("C:\\Users\\SONG\\Desktop\\CIAIC\\code\\ocr\\images\\da.jpg")
    cv2.namedWindow("input",cv2.WINDOW_AUTOSIZE)
    cv2.imshow("input",src)
    # b,g,r=cv2.split(src)
    # cv2.imshow("B",b)
    # cv2.imshow("G",g)
    # cv2.imshow("R",r)
    gray = cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
    cv2.imshow("GRAY",gray)

    ret,img = cv2.threshold(gray,120,255,cv2.THRESH_BINARY)####其中80适用于数码管数据  而且这个阈值可以通过直方图法或者平均法来进行自适应
    cv2.imshow("ret",img)

    turn = ~img            #####LCD屏不需要取反，即二值化之后原来就是白底黑字
    cv2.imshow("turn",turn)

    # # 提取图中红色部分
    # hsv = cv2.cvtColor(src,cv2.COLOR_BGR2HSV)
    # low_hsv = np.array([0,43,46])
    # high_hsv = np.array([10,255,255])
    # extract = cv2.inRange(hsv,lowerb=low_hsv,upperb=high_hsv)
    text = pytesseract.image_to_string(img, lang='eng',config="--psm 7")
    # text = pytesseract.image_to_string(image)
    print('chulihou:')
    print(text)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def OCR_lcd():
    lcd_image = cv2.imread("C:\\Users\\SONG\\Desktop\\CIAIC\\code\\ocr\\images\\obj1.jpg")
    cv2.namedWindow("input", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("input", lcd_image)

    gray = cv2.cvtColor(lcd_image, cv2.COLOR_BGR2GRAY)
    cv2.imshow("GRAY", gray)

    mean, stddev = cv2.meanStdDev(gray)
    # ret, image_bin = cv2.threshold(gray, mean[0][0]-10, 255, cv2.THRESH_BINARY)
    ret , image_bin = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    cv2.imshow("image_bin",image_bin)

    image_turn = ~image_bin
    image_show = image_turn
    cv2.imshow("image_turn",image_turn)

    contours, hierarchy = cv2.findContours(image_turn, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.imshow("image_find:",image_find)
    boundRect = []
    # bounds_ = []
    # boundRect_tran = []
    # print(contours)
    # print(np.shape(contours))
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        # 一个筛选，可能需要看识别条件而定，有待优化
        if h / w > 1:
            boundRect.append([x, y, w, h])
            # 画一个方形标注一下，看看圈的范围是否正确
            image_bound  = cv2.rectangle(image_turn, (x, y), (x+w, y+h), 255, 2)
    cv2.imshow("bound",image_bound)
    print("boundRect:",boundRect)
    print(np.shape(boundRect))
    #最好处理黑底白字
    text = pytesseract.image_to_string(image_turn, lang='eng', config='--psm 7')#lang='chi_sim'是设置为中文识别，
    print("lcd_text:",text)

    cv2.waitKey(0)
    cv2.destroyAllWindows()





if __name__ == '__main__':


    # Data = {'id':1}
    # # OCR_lcd()
    # serial_file_name = 'C:\\Users\\SONG\\Desktop\\2021-08\\1.jpg'
    # serial_image = split.detect_and_split_3(serial_file_name)
    # serial_list = []
    # serial_list.append(recognition.serial_recogn(serial_image))
    # Data.update(ultis.ser_list_to_dic(serial_list))
    #
    # lcd_file_name = 'C:\\Users\\SONG\\Desktop\\2021-08\\2.jpg'
    # lcd_image =  split.detect_and_split_2(lcd_file_name)
    # lcd_list = []
    # lcd_list.append(recognition.lcd_recogn('./images/test4.jpg'))
    # Data.update(ultis.lcd_list_to_dic(lcd_list[0]))
    #
    # digit_file_name = 'C:\\Users\\SONG\\Desktop\\2021-08\\3.jpg'
    # digit_image = split.detect_and_split_1(digit_file_name)
    # digit_list = []
    # for i in range(len(digit_image)):
    #     # cv2.imshow('obj_digit{}'.format(i),digit_image[i])
    #     digit_list.append(recognition.digit_detect(digit_image[i]))
    # Data.update(ultis.digit_list_to_dic(digit_list))
    #
    # print(Data)
    #
    # # Data.to_excel("./output/pvuv_pandas.xls", index=False)
    # file_name = 'C:\\Users\\SONG\\Desktop\\2021-08\\data.xlsx'
    # ultis.pd_toexcel(Data, file_name)
    # print('-----------------')
    # print('识别成功，数据已保存！')
    # print('-----------------')
    #
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


    # lcd_file_name = './images/lcd.png'
    # lcd_image = cv2.imread(lcd_file_name)
    # cv2.imshow('input',lcd_image)
    # lcd_img =[]
    # lcd_list = []
    # h = lcd_image.shape[0] // 4
    # for i in range(4):
    #     quarter = lcd_image[i * h:(i + 1) * h, 0:lcd_image.shape[1]]  # 这里要注意一下，后面需要把边框给去除
    #     lcd_img.append(quarter)
    #     cv2.imshow('obj_lcd{}'.format(i),lcd_img[i])
    #     lcd_list.append(recognition.lcd_detect(lcd_img[i]))
    # Data.update(ultis.lcd_list_to_dic(lcd_list))


    # lcd_file_name = './images/lcd.png'
    # lcd_image = split.detect_and_split_2(lcd_file_name)
    # # lcd_image = cv2.imread()
    # lcd_list = []
    # for i in range(len(lcd_image)):
    #     # cv2.imshow('obj_lcd{}'.format(i), lcd_image[i])
    #     # lcd_image[i] = cv2.morphologyEx(lcd_image[i], cv2.MORPH_OPEN, kernel,iterations=1)
    #     # cv2.imshow('obj_lcd{}'.format(i), lcd_image[i])
    #     lcd_list.append(recognition.lcd_detect(lcd_image[i]))
    # Data.update(ultis.lcd_list_to_dic(lcd_list))
    # print('lcd_list:',lcd_list)

    # digit_file_name = './images/4.jpg'
    # digit_image = split.detect_and_split_1(digit_file_name)
    # digit_list = []
    # for i in range(len(digit_image)):
    #     # cv2.imshow('obj_digit{}'.format(i),digit_image[i])
    #     digit_list.append(recognition.digit_detect(digit_image[i]))
    # Data.update(ultis.digit_list_to_dic(digit_list))

    file_name = 'E:\\desktop\\images2\\1-2.JPG'
    img = meas_point_1.find_squares(file_name)
    # img = split.find_squares(file_name)
    # img = cv2.imread(file_name)
    deng = meas_point_1.handcar_recog(img)
    print("last:",deng)

    # canny = cv2.Canny(gray,150,300)
    # cv2.namedWindow("canny", cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("canny", 1000, 750)
    # cv2.imshow("canny", canny)

    cv2.waitKey(0)
    cv2.destroyAllWindows()





