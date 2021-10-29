import cv2
import numpy as np


# 
# def #cvshow1000#(name, img):
#      cv2.namedWindow(name, cv2.WINDOW_NORMAL)
#      cv2.resizeWindow(name, 1000, 750)
#      cv2.imshow(name, img)

def gain_h_and_w(img):
    h = img.shape[0]
    w = img.shape[1]
    return h,w


# TODO： 把这个函数分成split和recog
def knife_switch_split_and_recog(image):
    img = image
    # 转到HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    print(hsv)

    # 设置阈值
    l_blue = np.array([[21, 43, 46]])
    h_blue = np.array([34, 255, 255])

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
    #cvshow1000#("bound",img)
    boundRect = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        # 一个筛选，可能需要看识别条件而定，有待优化
        if w/h < 1 and h > img.shape[0]//5:
            boundRect.append([x, y, w, h])
            # 画一个方形标注一下，看看圈的范围是否正确
            red_dil = cv2.rectangle(img, (x, y), (x + w, y + h), 255, 2)
            print(x, y, w, h)
            #cvshow1000#('bound', red_dil)
    if boundRect is not None:
        boundRect.sort(key=lambda x: x[0], reverse=False)
        print(boundRect)
        left_area = []
        right_area = []
        left_area = [boundRect[0][0] + boundRect[0][2] // 2 * 3, boundRect[0][1], boundRect[0][2], boundRect[0][3]]
        right_area = [boundRect[1][0] - boundRect[1][2] // 2 * 3, boundRect[1][1], boundRect[1][2], boundRect[1][3]]
        left_img = image_er[left_area[1]:left_area[1] + left_area[3], left_area[0]:left_area[0] + left_area[2]]
        right_img = image_er[right_area[1]:right_area[1] + right_area[3], right_area[0]:right_area[0] + right_area[2]]
        cv2.imshow("left_img", left_img)
        cv2.imshow("right_img", right_img)
        left_len_total = left_img.shape[0] * left_img.shape[1]
        left_len = len(left_img[left_img == 255])
        right_len_total = right_img.shape[0] * right_img.shape[1]
        right_len = len(right_img[right_img == 255])
        print(left_len_total, left_len)
        print(right_len_total, right_len)
        knife_switch_list = []
        if left_len < left_len_total // 15:
            knife_switch_list.append('下')
        else:
            knife_switch_list.append('上')
        if right_len < right_len_total // 15:
            knife_switch_list.append('下')
        else:
            knife_switch_list.append('上')
        print(knife_switch_list)  # 1表示黄条没有遮挡，所以开关向上，0表示黄条遮挡，开关向下。
        return knife_switch_list



def key_split(image):
    img = image
    # 转到HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # 设置阈值
    l_blue = np.array([[21, 43, 46]])
    h_blue = np.array([34, 255, 255])
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
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        # 一个筛选，可能需要看识别条件而定，有待优化
        if  h < img.shape[0]//2 and w/h > 2.5  and h>20:
            boundRect.append([x, y, w, h])
            # 画一个方形标注一下，看看圈的范围是否正确
            red_dil = cv2.rectangle(img, (x, y), (x + w, y + h), 255, 2)
            print(x, y, w, h)
            #cvshow1000#('red_dir', red_dil)
    if boundRect is not None:
        boundRect.sort(key=lambda x: x[1], reverse=False)
        print(boundRect)
        img_up_area = [boundRect[0][0] - boundRect[0][2] * 8, boundRect[0][1] - boundRect[0][2] * 2,
                       boundRect[0][2] * 17, boundRect[0][2] * 2]
        img_up_img = img[img_up_area[1]:img_up_area[1] + img_up_area[3], img_up_area[0]:img_up_area[0] + img_up_area[2]]
        img_down_area = [boundRect[1][0] - boundRect[1][2] * 8, boundRect[1][1] - boundRect[1][2] * 2,
                         boundRect[1][2] * 17, boundRect[1][2] * 2]
        img_down_img = img[img_down_area[1]:img_down_area[1] + img_down_area[3],
                       img_down_area[0]:img_down_area[0] + img_down_area[2]]
        cv2.imshow("up", img_up_img)
        cv2.imshow("down", img_down_img)
        return img_up_img, img_down_img




def key_recog(img):
    h,w = img.shape[0], img.shape[1]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(3,3),0)
    cv2.imshow("gray2", gray)
    canny = cv2.Canny(gray,150,300)
    cv2.imshow("canny",canny)
    ret, image_er = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)  # OTSU最大类间方差法
    cv2.imshow("er_recog",image_er)
    th2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 5)
    cv2.imshow("th2",th2)
    img_dis = []
    lines = []
    key_list = []
    k_sum = []
    # cv2.imshow("dis",img_dis)
    for i in range(5):
        img_dis.append(th2[ h//4:h//4*3, i*w//5:(i+1)*w//5 ])
        lines = cv2.HoughLinesP(img_dis[i], 1, np.pi / 180, threshold=20, minLineLength=130)
        # cv2.imshow("dis",img_dis)
        cv2.imshow('obj{}'.format(i),img_dis[i])
        k = []
        if lines is not None:
            for x1, y1, x2, y2 in lines[:, 0]:
                cv2.line(img, (x1 + i * w // 5, y1 + h // 4), (x2 + i * w // 5, y2 + h // 4), (0, 0, 255), 2)
                k.append((y1 + y2) / 2 + h // 4)
                print("line", lines)
            cv2.imshow("line",img)
            # print(sum(k), len(k))
            # k_sum.append(sum(k)/len(k))
            if sum(k) / len(k) > h // 2:
                key_list.append('下')
            else:
                key_list.append('上')

    # print(k_sum)
    print(key_list)
    return key_list

def key_recogn2(img):
    h, w = img.shape[0], img.shape[1]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    cv2.imshow("gray2", gray)
    canny = cv2.Canny(gray, 150, 300)
    cv2.imshow("cnany", canny)
    ret, image_er = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)  # OTSU最大类间方差法
    cv2.imshow("er_recog", image_er)
    th2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 5)
    cv2.imshow("th", th2)
    img_dis = []
    lines = []
    key_list = []
    k_sum = []
    # cv2.imshow("dis",img_dis)
    for i in range(5):
        img_up = img[0:h//2, i * w // 5:(i + 1) * w // 5]
        img_down = img[h//2:h, i * w // 5:(i + 1) * w // 5]
        len_up = len(img_up[img_up == 0])
        print("len_up", len_up)
        len_down = len(img_down[img_down == 0])
        print("len_down", len_down)
        if len_up>len_down:
            key_list.append(1)
        else:
            key_list.append(0)

    # print(k_sum)
    print(key_list)
    return key_list

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
        cv2.imshow("up", img_up)
        cv2.imshow("down", img_down)

        return img_up, img_down

def instruct_led_recog(img):
    h,w = img.shape[0], img.shape[1]
    led_list = []
    for i in range(8):
        image_zhong = img[:,i*w//8:(i+1)*w//8]
        # cv2.imshow("image_yuan", image_yuan)
        hsv = cv2.cvtColor(image_zhong, cv2.COLOR_RGB2HSV)
        H, S, V = cv2.split(hsv)
        # print("HSV", H, S, V)
        v = V.ravel()[np.flatnonzero(V)]  # 亮度非零的值
        average_v = sum(v) / len(v)
        print("average_v", average_v)
        if average_v < 165:
            led_list.insert(i,'灭')
        else:
            led_list.insert(i,'亮')
    print("led_list",led_list)
    return led_list

def getcode_area(img, code_info):
    code_area = np.array(code_info.points)
    maxindex = np.argmax(code_info.points, axis=0)
    minindex = np.argmin(code_info.points, axis=0)
    x_min = int(code_area[minindex[0]][0])
    y_min = int(code_area[minindex[1]][1])
    x_max = int(code_area[maxindex[0]][0])
    y_max = int(code_area[maxindex[1]][1])
    w = x_max - x_min
    h = y_max - y_min
    code_img = img[y_min - h // 4:y_max + h // 4, x_min - w // 4:x_max + w // 4]
    cv2.imshow("code_img", code_img)
    code_location = [x_min - w // 4, y_min - h // 4, x_max - x_min + w // 2, y_max - y_min + h // 2]
    print("getcode_area:", code_location)
    return code_img, code_location

def instruct_led_split(img, code_location):
    img_down_area = [code_location[0] - code_location[2]//10*12, code_location[1] - code_location[3]//2*3,
                     code_location[2]//10*34, code_location[3]//2 ]
    img_down = img[img_down_area[1]:img_down_area[1]+img_down_area[3], img_down_area[0]:img_down_area[0]+img_down_area[2]]
    cv2.imshow("down", img_down)
    img_up_area = [code_location[0] - code_location[2]//10*12, code_location[1] - code_location[3]//10*32,
                     code_location[2]//10*34, code_location[3]//2 ]
    img_up = img[img_up_area[1]:img_up_area[1]+img_up_area[3], img_up_area[0]:img_up_area[0]+img_up_area[2]]
    cv2.imshow("up", img_up)
    return img_up, img_down

def judge_pic(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # 色彩空间转换为hsv，便于分离
    lower_hsv1 = np.array([156, 43, 46])  # 提取颜色的低值 red [156, 43, 46]
    high_hsv1 = np.array([180, 255, 255])  # 提取颜色的高值 [180, 255, 255]
    lower_hsv2 = np.array([0, 43, 46])  # 提取颜色的低值 red [0, 43, 46]
    high_hsv2 = np.array([10, 255, 255])  # 提取颜色的高值 [10, 255, 255]
    mask_1 = cv2.inRange(hsv, lowerb=lower_hsv1, upperb=high_hsv1)
    mask_2 = cv2.inRange(hsv, lowerb=lower_hsv2, upperb=high_hsv2)
    mask = cv2.bitwise_or(mask_1, mask_2)
    # gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    ret, image_er = cv2.threshold(mask, 0, 255, cv2.THRESH_OTSU)  # OTSU最大类间方差法
    len1 = len(mask[mask==255])
    len2 = len(mask[mask==0])
    len3 = mask.shape[0]*mask.shape[1]
    print(len1, len2, len3)
    return len1

def judge_pic2(img, code_location):
    print(img.shape[0]//3*2)
    print(code_location[1])
    if code_location[1] >= img.shape[0]//3*2:
        return '2'
    else:
        return '1'

def tran(list):
    list_str = str(list).replace("[", "").replace("]", "")
    return eval(f"[{list_str}]")

#
# def Point_twenty_five(file_name1, file_name2, file_name3):
#     img1 = cv2.imread(file_name1)
#     img2 = cv2.imread(file_name2)
#     img3 = cv2.imread(file_name3)
#     Point_list = []
#     Point_list.append(knife_switch_split_and_recog(img2))
#     img_up, img_down = key_split(img3)
#     Point_list.append(key_recog(img_up))
#     Point_list.append(key_recog(img_down))
#
#     print("point_twenty_six:", Point_list)
#     return tran(Point_list)


def Point_twenty_five(img_path, code_info):
    img = cv2.imread(img_path)
    img2 = img[:, img.shape[1] // 4:img.shape[1] // 4 * 3]
    Point_list = []
    _, code_location = getcode_area(img2, code_info)
    #判断是第几张照片
    # judge_pic(img2)
    Point_list.append(judge_pic2(img2, code_location))
    if Point_list[0] == '2':
        instruct_led_img_up, instruct_led_img_down = instruct_led_split(img2, code_location)

        Point_list.append(instruct_led_recog(instruct_led_img_up))
        Point_list.append(instruct_led_recog(instruct_led_img_down))
    if Point_list[0] == '1':
        list = [' '] * 5
        Point_list.append(list)

    # instruct_led_img_up, instruct_led_img_down = instruct_led_split(img2, code_location)
    #
    # Point_list.append(instruct_led_recog(instruct_led_img_up))
    # Point_list.append(instruct_led_recog(instruct_led_img_down))

    # instruct_led_up_img, instruct_led_down_img = instruct_led_split2(img2)
    # Point_list.append(instruct_led_recog(instruct_led_up_img))
    # Point_list.append(instruct_led_recog(instruct_led_down_img))
    # # Point_list.append([1, 0])
    # print("Point_twenty_seven", Point_list)
    return tran(Point_list)


