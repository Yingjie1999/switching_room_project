import copy

import cv2
import numpy as np
import code_recog

#将断路器的分割与识别封装成一个类

class Breaker(object):
    def __init__(self, img, code_info):
        self.img = img
        self.code_info = code_info
        self.result = []

    def gain_h_and_w(self, img):
        h = img.shape[0]
        w = img.shape[1]
        return h, w

    def getcode_area(self):
        code_area = np.array(self.code_info.points)
        maxindex = np.argmax(self.code_info.points, axis=0)
        minindex = np.argmin(self.code_info.points, axis=0)
        x_min = int(code_area[minindex[0]][0])
        y_min = int(code_area[minindex[1]][1])
        x_max = int(code_area[maxindex[0]][0])
        y_max = int(code_area[maxindex[1]][1])
        w = x_max - x_min
        h = y_max - y_min
        code_img = self.img[y_min - h // 4:y_max + h // 4, x_min - w // 4:x_max + w // 4]
        cv2.imshow("code_img", code_img)
        code_location = [x_min - w // 4, y_min - h // 4, x_max - x_min + w // 2, y_max - y_min + h // 2]
        print("getcode_area:", code_location)
        return code_img, code_location

    def L1toL3Led_split(self):
        image = self.img
        image_h, image_w = self.gain_h_and_w(image)
        image_split = image[image_h//2:image_h, 0:image_w//2]
        gray = cv2.cvtColor(image_split, cv2.COLOR_BGR2GRAY)
        # cv2.imshow("gray", gray)
        ret, image_er = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)  # OTSU最大类间方差法
        # cv2.namedWindow#("er", cv2.WINDOW_NORMAL)
        # cv2.resizeWindow#("er", 1000, 750)
        cv2.imshow("er", image_er)
        contours, hierarchy = cv2.findContours(image_er, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 找轮廓
        # cv2.drawContours(image_split, contours, -1, (0, 255, 0), 3)  # 把边缘给画出来
        boundRect = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            # 一个筛选，可能需要看识别条件而定，有待优化
            if w / h > 1 and w < 0.4 * image_w and w > 0.05 * image_w  and h < 0.4 * image_h and h > 0.05 * image_h and x > 10 and y > 10:
                boundRect.append([x, y, w, h])
                # 画一个方形标注一下，看看圈的范围是否正确
                # red_dil = cv2.rectangle(image_split, (x, y), (x + w, y + h), 255, 2)
                # # print(x, y, w, h)
        print(boundRect)
        if boundRect is not None:
            # print(np.shape(boundRect))
            # 暂时通过最大值来判断
            a = np.array(boundRect)
            maxindex = a.argmax(axis=0)
            print(maxindex)
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
            cv2.imshow("lll_img", image_out)
            return image_out, led_coordinate

    # 将远方开关和预合开关给分割出来
    def APTkey_split(self, L1toL3Led_location):
        image = self.img
        image_h, image_w = self.gain_h_and_w(image)
        image_split = image[image_h // 2 + L1toL3Led_location[1]:image_h // 2 + L1toL3Led_location[1] + L1toL3Led_location[3] * 2,
                      L1toL3Led_location[0] + L1toL3Led_location[2]:L1toL3Led_location[0] + L1toL3Led_location[2] * 5 // 2]
        # cv2.imshow("image_split", image_split)
        gray = cv2.cvtColor(image_split, cv2.COLOR_BGR2GRAY)
        # cv2.imshow("gray", gray)
        ret, image_er = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)  # OTSU最大类间方差法
        contours, hierarchy = cv2.findContours(image_er, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 找轮廓
        # cv2.drawContours(image_split, contours, -1, (0, 255, 0), 3)  # 把边缘给画出来
        # cv2.imshow("contour",image_split)
        split_h, split_w = self.gain_h_and_w(image_split)
        boundRect = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            # 一个筛选，可能需要看识别条件而定，有待优化
            if w / h < 1.2 and w / h > 0.8 and w < 0.8 * split_w and h < 0.8 * split_h and h > 0.1 * split_h:
                boundRect.append([x, y, w, h])
                # 画一个方形标注一下，看看圈的范围是否正确
                red_dil = cv2.rectangle(image_split, (x, y), (x + w, y + h), 255, 2)
                cv2.imshow('bound', red_dil)
        print(boundRect)
        if boundRect is not None:
            boundRect.sort(key=lambda x: x[3], reverse=True)#取最大的两个框
            print(boundRect)
            # if boundRect[0][0]<=boundRect[1][0]:
            # a = np.array(boundRect)
            # maxindex = a.argmax(axis=-1)
            # print(maxindex)
            # b = np.delete(a, 0, axis=0)
            # maxindex_second = b.argmax(axis=-1)
            # # print(b[1])
            # if 0 <= 1:
            #     maxindex_second = [i + 0 for i in maxindex_second]
            # 比较两个开关的位置
            if boundRect[0][0] <= boundRect[1][0]:
                image_return_yuanfang = image_split[
                                        boundRect[0][1]:boundRect[0][1] + boundRect[0][3],
                                        boundRect[0][0]:boundRect[0][0] + boundRect[0][2]]
                image_return_yuhe = image_split[boundRect[1][1]:boundRect[1][1] + boundRect[1][3],
                                    boundRect[1][0]:boundRect[1][0] + boundRect[1][2]]
                return image_return_yuanfang, image_return_yuhe
            else:
                image_return_yuanfang = image_split[boundRect[1][1]:boundRect[1][1] + boundRect[1][3],
                                        boundRect[1][0]:boundRect[1][0] + boundRect[1][2]]
                image_return_yuhe = image_split[
                                    boundRect[0][1]:boundRect[0][1] + boundRect[0][3],
                                    boundRect[0][0]:boundRect[0][0] + boundRect[0][2]]
                return image_return_yuanfang, image_return_yuhe

    def lianpian_split(self, code_location):
        lianpian_area = [code_location[0] + code_location[2] * 3, code_location[1] + code_location[3] // 2 * 3,
                         code_location[2] // 4 * 5,code_location[3] // 4 * 3]
        lianpian_img = self.img[lianpian_area[1]:lianpian_area[1] + lianpian_area[3],
                       lianpian_area[0]:lianpian_area[0] + lianpian_area[2]]
        cv2.imshow("lp_img", lianpian_img)
        return lianpian_img, lianpian_area

    def lighting_storage_key_split(self, code_location):
        b_w_key_area = [code_location[0] + code_location[2] * 3, code_location[1] + code_location[3],
                        code_location[2] // 4 * 5, code_location[3] // 2]
        b_w_key_img = self.img[b_w_key_area[1]:b_w_key_area[1] + b_w_key_area[3],
                      b_w_key_area[0]:b_w_key_area[0] + b_w_key_area[2]]
        cv2.imshow("ls_img", b_w_key_img)
        return b_w_key_img, b_w_key_area

    def indicator_light_split(self, code_location):
        led_location = [code_location[0] + code_location[2] * 3, code_location[1] + code_location[3] // 2,
                        code_location[2] // 4 * 5, code_location[3] // 2]
        led_img = self.img[led_location[1]:led_location[1] + led_location[3],
                  led_location[0]:led_location[0] + led_location[2]]
        cv2.imshow("il_img", led_img)
        return led_img, led_location

    def position_indicator_split(self, code_location):
        handcar_location = [code_location[0] + code_location[2] * 3, code_location[1], code_location[2] // 4 * 5, code_location[3] // 2]
        led_img = self.img[handcar_location[1]:handcar_location[1] + handcar_location[3],
                  handcar_location[0]:handcar_location[0] + handcar_location[2]]
        cv2.imshow("hand_img", led_img)
        return led_img, handcar_location

    def  running_led_split(self, code_location):
        running_location = [code_location[0] + code_location[2] // 4 * 9, code_location[1], code_location[2] // 2, code_location[3]]
        running_led_img = self.img[running_location[1]:running_location[1] + running_location[3],
                  running_location[0]:running_location[0] + running_location[2]]
        cv2.imshow("rl_img", running_led_img)
        return running_led_img, running_location

    def L1toL3led_recog(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 3)
        # cv2.imshow("gray", gray)
        ret, image_er = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)  # OTSU最大类间方差法
        canny = cv2.Canny(image_er, 50, 150)
        circles = cv2.HoughCircles(canny, cv2.HOUGH_GRADIENT, 1, 35, param1=300, param2=20, minRadius=0,
                                   maxRadius=image_er.shape[0] // 4)
        led_list = ['灭', '灭', '灭']
        if circles is not None:
            circles = np.uint16(np.around(circles))  # around对数据四舍五入，为整数
            for i in circles[0, :]:
                cv2.circle(image, (i[0], i[1]), i[2], (0, 0, 255), 2)  # 画圆
                cv2.circle(image, (i[0], i[1]), 2, (0, 0, 255), 2)  # 画圆心
            print(circles)

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

                if circles[0][i][0] > image.shape[1] // 5 * 2 and circles[0][i][0] < image.shape[
                    1] // 5 * 3 and average_v > 120:
                    led_list[1] = '亮'

                if circles[0][i][0] >= image.shape[1] // 5 * 3 and average_v > 120:
                    led_list[2] = '亮'

            print(led_list)
            if len(led_list) == 3:
                return led_list
            else:
                return [' ', ' ', ' ']
        else:
            return [' ', ' ', ' ']

    def APTkey_yuanfang_recogn(self, image):
        h, w = self.gain_h_and_w(image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        ret, image_er = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)  # OTSU最大类间方差法
        cv2.imshow("img_er", image_er)
        img_down = image_er[h // 4 * 3:h, 0:w]
        # cv2.imshow("down",img_down)
        img_right = image_er[0:h, w // 4 * 3:w]
        # cv2.imshow("right",img_right)
        len_down1 = len(img_down[img_down == 0])
        # len_down2 = len(img_down[img_down==255])
        len_right1 = len(img_right[img_right == 0])
        # len_right2 = len(img_right[img_right==255])
        # 0表示开关为竖着的状态（远地开关），1表示横着的状态（就地开关）
        if len_down1 > len_right1:
            return '远方'
        else:
            return '就地'


    def APTkey_yuhe_recogn(self, image):

        h, w = self.gain_h_and_w(image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        ret, image_er = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)  # OTSU最大类间方差法
        cv2.imshow("img_er", image_er)
        img_down = image_er[h // 4 * 3:h, 0:w]
        # cv2.imshow("down",img_down)
        img_right = image_er[0:h, w // 4 * 3:w]
        # cv2.imshow("right",img_right)
        len_down1 = len(img_down[img_down == 0])
        # len_down2 = len(img_down[img_down==255])
        len_right1 = len(img_right[img_right == 0])
        # len_right2 = len(img_right[img_right==255])
        # 0表示开关为竖着的状态（远地开关），1表示横着的状态（就地开关）
        if len_down1 > len_right1:
            return '预合合后'
        else:
            return '预合分后'

    def lianpian_recog(self, image):
        img = image
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        l_yellow = np.array([[14, 43, 46]])
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
        cv2.imshow("mask_yellow", mask_yellow)
        cv2.imshow("mask_red", mask_red)
        cv2.imshow("mask", mask)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 找轮廓
        # cv2.drawContours(img, contours, -1, (0, 255, 0), 3)  # 把边缘给画出来
        # cv2.imshow("bound",img)
        boundRect = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            # 一个筛选，可能需要看识别条件而定，有待优化
            if h < img.shape[0] * 0.95 and w / h < 2 and w / h > 0.1 and h > img.shape[0] // 3:
                boundRect.append([x, y, w, h])
                # 画一个方形标注一下，看看圈的范围是否正确
                red_dil = cv2.rectangle(img, (x, y), (x + w, y + h), 255, 2)
                cv2.imshow("lianpian_red", red_dil)
        print(boundRect)

        if boundRect is not None:
            boundRect.sort(key=lambda x: x[0], reverse=False)
            lianpian_list = []
            for i in range(len(boundRect)):
                if boundRect[i][2] < boundRect[i][3] * 0.7:
                    lianpian_list.insert(i, '合')  # 0代表闭合，1代表打开
                else:
                    lianpian_list.insert(i, '分')
                print(boundRect[i][2] / boundRect[i][3])
            print("lianpian:", lianpian_list)
            if len(lianpian_list) != 2:
                lianpian_list = [' ', ' ']
            return lianpian_list
        else:
            lianpian_list = [' ', ' ']
            return lianpian_list

    def lighting_storage_key_recog(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 3)
        ret, image_er = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)  # OTSU最大类间方差法
        circles = cv2.HoughCircles(image_er, cv2.HOUGH_GRADIENT, 1, 50, param1=200, param2=28, minRadius=20,
                                   maxRadius=image_er.shape[0] // 2)
        # circles = np.array([])
        if circles is not None:
            circles = np.uint16(np.around(circles))  # around对数据四舍五入，为整数
            for i in circles[0, :]:
                cv2.circle(img, (i[0], i[1]), i[2], (0, 0, 255), 2)  # 画圆
                # cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 2)  # 画圆心
            cv2.imshow("b_key_yuan", img)
            print(circles)
            cir = []
            for i in range(len(circles[0])):
                cir.append(circles[0][i])
            print(cir[0])
            cir.sort(key=lambda x: x[0], reverse=False)
            print(len(cir))
            cv2.imshow("yuan", img)
            k = []
            k_sum = []
            key_list = []
            for i in range(len(cir)):
                image_zhong = img[cir[i][1] - cir[i][2]:cir[i][1] + cir[i][2],
                              cir[i][0] - cir[i][2]:cir[i][0] + cir[i][2]]
                image_yuan = gray[cir[i][1] - cir[i][2]:cir[i][1] + cir[i][2],
                             cir[i][0] - cir[i][2]:cir[i][0] + circles[0][i][2]]
                # cv2.imshow("image_yuan", image_yuan)
                edge = cv2.Canny(image_yuan, 50, 150)
                # cv2.imshow("bound", edge)
                lines = cv2.HoughLinesP(edge, 1, np.pi / 180, threshold=30, minLineLength=10)
                if lines is not None:
                    for x1, y1, x2, y2 in lines[:, 0]:
                        cv2.line(image_zhong, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    # print("line", lines)
                    cv2.imshow("xian,", image_zhong)
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
            print("key_list", key_list)
            if len(key_list) != 2:
                key_list = [' ', ' ']
            return key_list
        else:
            key_list = [' ', ' ']
            return key_list
    def indicator_led_recog(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 3)
        # gray = cv2.GaussianBlur(gray,(3,3),0)
        # cv2.imshow("gray", gray)
        canny = cv2.Canny(gray, 50, 150)
        cv2.imshow("led_canny", canny)
        ret, image_er = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)  # OTSU最大类间方差法
        cv2.imshow("led_er", image_er)
        circles = cv2.HoughCircles(canny, cv2.HOUGH_GRADIENT, 1, 50, param1=150, param2=28, minRadius=20,
                                   maxRadius=image_er.shape[0] // 4)
        # circles = np.array([])
        if circles is not None:
            circles = np.uint16(np.around(circles))  # around对数据四舍五入，为整数
            # leng = len(circles[0][0])
            # for i in range(leng):
            #     if circles[0][0][i][0] < image_split.shape[0] // 8:
            #         # print("i:",circles[0][0][i][0])
            #         circles = np.delete(circles[0][0], i, 0)

            for i in circles[0, :]:
                cv2.circle(img, (i[0], i[1]), i[2], (255, 0, 0), 2)  # 画圆
                cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 2)  # 画圆心
            print(circles)
            cv2.imshow("sangeyuan", img)

            cir = []
            for i in range(len(circles[0])):
                cir.append(circles[0][i])
            print(cir[0])
            cir.sort(key=lambda x: x[0], reverse=False)
            print(len(cir))
            cv2.imshow("yuan", img)
            led_list = []
            for i in range(len(cir)):
                image_zhong = img[cir[i][1] - cir[i][2]:cir[i][1] + cir[i][2],
                              cir[i][0] - cir[i][2]:cir[i][0] + cir[i][2]]
                # cv2.imshow("image_yuan", image_yuan)
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
            if len(led_list) != 3:
                led_list = [' ', ' ', ' ']
            return led_list
        else:
            return [' ', ' ', ' ']

    def position_indicator_led_recog(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        cv2.imshow("gray", gray)
        canny = cv2.Canny(gray, 50, 150)
        cv2.imshow("canny", canny)
        ret, image_er = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)  # OTSU最大类间方差法
        cv2.imshow("handcar_er", image_er)
        contours, hierarchy = cv2.findContours(image_er, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 找轮廓
        boundRect = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            # 一个筛选，可能需要看识别条件而定，有待优化
            if h > img.shape[0] // 3 and h < img.shape[0] * 0.95 and w / h < 1.12 and w / h > 0.82:
                boundRect.append([x, y, w, h])
                # 画一个方形标注一下，看看圈的范围是否正确
                # red_dil = cv2.rectangle(img, (x, y), (x + w, y + h), 255, 2)
                # cv2.imshow("red",red_dil)
        # cv2.imshow("min", img)
        if boundRect is not None:
            boundRect.sort(key=lambda x: x[0], reverse=False)
            print("boundRect", boundRect)
            boundRect_ = copy.deepcopy(boundRect)
            for i in range(len(boundRect) - 1):
                if boundRect[i][0] + boundRect[i][2] >= boundRect[i+1][0] + boundRect[i+1][2]:
                    boundRect_.pop(i)
            print(boundRect_)
            # w_average = np.mean(boundRect, axis=0)[2]
            # print(w_average)
            # for i in range(len(boundRect)):
            #     if len(boundRect) == 3:
            #         boundRect_.append(boundRect[i])
            #     if boundRect[i][2] > w_average:
            #         boundRect_.append(boundRect[i])
            # print(boundRect_)
            indicator_list = []
            for i in range(len(boundRect_)):
                img_quar = img[boundRect_[i][1]:boundRect_[i][1] + boundRect_[i][3],
                           boundRect_[i][0]:boundRect_[i][0] + boundRect_[i][2]]
                hsv = cv2.cvtColor(img_quar, cv2.COLOR_BGR2HSV)
                # H,S,V = np.split(hsv)
                H, S, V = cv2.split(hsv)
                # print("HSV", H, S, V)
                v = V.ravel()[np.flatnonzero(V)]  # 亮度非零的值
                average_v = sum(v) / len(v)
                print("average_v", average_v)
                hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
                object_H = np.argmax(hist)
                print("object_H:", object_H)

                if object_H >= 156 and object_H <= 180 and average_v > 120:
                    indicator_list.append("合闸")
                elif object_H >= 0 and object_H < 10 and average_v > 120:
                    indicator_list.append("合闸")
                elif object_H > 10 and object_H <= 50 and average_v > 120:
                    indicator_list.append("分闸")
                else:
                    indicator_list.append(" ")
            print("indicator_list:", indicator_list)
            if len(indicator_list) != 3:
                indicator_list = [' ', ' ', ' ']
            return indicator_list
        else:
            indicator_list = [' ', ' ', ' ']
            return indicator_list

    def running_led_recog(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray,(3,3),0)
        cv2.imshow("gray", gray)
        ret, image_er = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)  # OTSU最大类间方差法
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        th = cv2.erode(image_er,kernel)
        cv2.imshow("dilate", th)
        contours, hierarchy = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 找轮廓
        # cv2.drawContours(img, contours, -1, (0, 255, 0), 3)  # 把边缘给画出来
        # cv2.imshow("bianyuan",img)
        boundRect = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            # 一个筛选，可能需要看识别条件而定，有待优化
            if h < img.shape[0] * 0.95 and w / h > 2.5 and w > img.shape[1] // 5 and w / h < 5:
                boundRect.append([x, y, w, h])
                # 画一个方形标注一下，看看圈的范围是否正确
                red_dil = cv2.rectangle(img, (x, y), (x + w // 2, y + h), 255, 2)
                cv2.imshow("yunixng_led", red_dil)
        if boundRect is not None:
            boundRect = list(set([tuple(t) for t in boundRect]))
            boundRect.sort(key=lambda x: x[1], reverse=False)
            print(boundRect)
            color = []
            led_list = []
            for i in range(len(boundRect)):
                img_quar = img[boundRect[i][1]:boundRect[i][1] + boundRect[i][3],
                           boundRect[i][0]:boundRect[i][0] + boundRect[i][2] // 2]
                hsv = cv2.cvtColor(img_quar, cv2.COLOR_BGR2HSV)
                H, S, V = cv2.split(hsv)
                # print("HSV", H, S, V)
                v = V.ravel()[np.flatnonzero(V)]  # 亮度非零的值
                average_v = sum(v) / len(v)
                print("average_v", average_v)
                if average_v < 125:
                    led_list.insert(i, '灭')
                else:
                    led_list.insert(i, '亮')
            print("led_list", led_list)
            if len(led_list) != 6:
                led_list = [' ', ' ', ' ', ' ', ' ', ' ']
            return led_list
        else:
            led_list = ['灭', '灭', '灭', '灭', '灭', '灭']
            return led_list

    def split_and_recog(self):
        _, code_location = self.getcode_area()
        L1toL3Led_img, L1toL3Led_location = self.L1toL3Led_split()
        self.result.append(self.L1toL3led_recog(L1toL3Led_img))
        yuanfangimg, yuhe_img = self.APTkey_split(L1toL3Led_location)
        self.result.append(self.APTkey_yuanfang_recogn(yuanfangimg))
        self.result.append(self.APTkey_yuhe_recogn(yuhe_img))
        lianpian_img, _ = self.lianpian_split(code_location)
        self.result.append(self.lianpian_recog(lianpian_img))
        lighting_storage_img, _ = self.lighting_storage_key_split(code_location)
        self.result.append(self.lighting_storage_key_recog(lighting_storage_img))
        indicator_led_img, _ = self.indicator_light_split(code_location)
        self.result.append(self.indicator_led_recog(indicator_led_img))
        position_led_img, _ = self.position_indicator_split(code_location)
        self.result.append(self.position_indicator_led_recog(position_led_img))
        running_led_img, _ = self.running_led_split(code_location)
        self.result.append(self.running_led_recog(running_led_img))

        print("LIST:", self.result)

if __name__ == "__main__":
    file_path = 'C:\\Users\\SONG\\Desktop\\image6\\22_14_28_770.jpg'
    img = cv2.imread(file_path)
    img2 = img[:, img.shape[1] // 4:img.shape[1] // 4 * 3]
    code_info = code_recog.ocr_qrcode_zxing(file_path)
    print(code_info)
    Point = Breaker(img2, code_info)
    Point.split_and_recog()

    cv2.waitKey(0)
    cv2.destroyAllWindows()
