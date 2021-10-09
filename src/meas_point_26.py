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


def instruct_led_recogn(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)
    # gray = cv2.GaussianBlur(gray,(3,3),0)
    # #cv2.imshow#("gray", gray)
    ret, image_er = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)  # OTSU最大类间方差法
    #cv2.imshow#("led_er",image_er)
    circles = cv2.HoughCircles(image_er, cv2.HOUGH_GRADIENT, 1, 200, param1=150, param2=40, minRadius=50,
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
        #cv2.imshow#("sangeyuan", img)

        cir = []
        for i in range(len(circles[0])):
            cir.append(circles[0][i])
        print(cir[0])
        cir.sort(key=lambda x: x[0], reverse=False)
        print(len(cir))
        #cv2.imshow#("yuan", img)
        led_list = []
        for i in range(len(cir)):
            image_zhong = img[cir[i][1] - cir[i][2]:cir[i][1] + cir[i][2],
                          cir[i][0] - cir[i][2]:cir[i][0] + cir[i][2]]
            # #cv2.imshow#("image_yuan", image_yuan)
            hsv = cv2.cvtColor(image_zhong, cv2.COLOR_RGB2HSV)
            H, S, V = cv2.split(hsv)
            # print("HSV", H, S, V)
            v = V.ravel()[np.flatnonzero(V)]  # 亮度非零的值
            average_v = sum(v) / len(v)
            print("average_v", average_v)
            if average_v < 150:
                led_list.insert(i, 0)
            else:
                led_list.insert(i, 1)
        print("led_list", led_list)
        return led_list

def instruct_led_split_and_recog(image):
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
        if  h > img.shape[0] // 50:
            boundRect.append([x, y, w, h])
            # 画一个方形标注一下，看看圈的范围是否正确
            red_dil = cv2.rectangle(img, (x, y), (x + w, y + h), 255, 2)
            print(x, y, w, h)
            #cvshow1000#('bound', red_dil)
    boundRect.sort(key=lambda x: x[2], reverse=True)
    print(boundRect)
    led_list = []
    for i in range(2):
        image_zhong = img[boundRect[i][1]:boundRect[i][1]+boundRect[i][3], boundRect[i][0]:boundRect[i][0]+boundRect[i][2]]
        # #cv2.imshow#("image_yuan", image_yuan)
        hsv = cv2.cvtColor(image_zhong, cv2.COLOR_RGB2HSV)
        H, S, V = cv2.split(hsv)
        # print("HSV", H, S, V)
        v = V.ravel()[np.flatnonzero(V)]  # 亮度非零的值
        average_v = sum(v) / len(v)
        print("average_v", average_v)
        if average_v < 150:
            led_list.insert(i, '灭')
        else:
            led_list.insert(i, '亮')
    print("led_list", led_list)
    boundRect_return = []
    if boundRect[0][0] > boundRect[1][0]:
        boundRect_return.append(boundRect[1])
        boundRect_return.append(boundRect[0])
    else:
        boundRect_return.append(boundRect[0])
        boundRect_return.append(boundRect[1])
    return led_list, boundRect_return

def APTkey_split(image, last_area):
    img = image
    h, w = img.shape[0], img.shape[1]
    APTkey_area = [last_area[1][0]*2 - last_area[0][0] - last_area[0][3], last_area[1][1]-last_area[1][3],
                   last_area[1][2]*3,last_area[1][3]*3]
    APTkey_img = img[APTkey_area[1]:APTkey_area[1]+APTkey_area[3],APTkey_area[0]:APTkey_area[0]+APTkey_area[2]]
    #cv2.imshow#("APT",APTkey_img)
    return APTkey_img

def APTkey_recogn(img):

    h, w = img.shape[0], img.shape[1]
    # print(h,w)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    ret, image_er = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)  # OTSU最大类间方差法
    #cv2.imshow#("img_er",image_er)
    contours, hierarchy = cv2.findContours(image_er, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 找轮廓
    # cv2.drawContours(img, contours, -1, (0, 255, 0), 3)  # 把边缘给画出来
    # #cv2.imshow#("luokuo", img)
    boundRect = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        # 一个筛选，可能需要看识别条件而定，有待优化
        if  h > img.shape[0] // 10 and h < 0.9*img.shape[0] and w>img.shape[1]//10  and w / h >0.5 and w / h <1.5:
            boundRect.append([x, y, w, h])
            # 画一个方形标注一下，看看圈的范围是否正确
            red_dil = cv2.rectangle(img, (x, y), (x + w, y + h), 255, 2)
            # print(x, y, w, h)
            #cv2.imshow#('red_dir', red_dil)
    boundRect.sort(key=lambda x: x[2], reverse=True)
    print(boundRect)
    img_left = image_er[boundRect[0][1]+boundRect[0][3]//2:boundRect[0][1]+boundRect[0][3],
                        boundRect[0][0]:boundRect[0][0]+boundRect[0][2]//2]
    # #cv2.imshow#("down",img_down)
    img_right = image_er[boundRect[0][1]+boundRect[0][3]//2:boundRect[0][1]+boundRect[0][3],
                        boundRect[0][0]+boundRect[0][2]//2:boundRect[0][0]+boundRect[0][2]]
    # #cv2.imshow#("right",img_right)
    len_left = len(img_left[img_left==0])
    # len_down2 = len(img_down[img_down==255])
    len_right = len(img_right[img_right==0])
    # len_right2 = len(img_right[img_right==255])
    #0表示开关为竖着的状态（远地开关），1表示横着的状态（就地开关）
    if len_left>len_right:
        return '2#'
    else:
        return '1#'

def running_led_split(image):
    img = image
    img_split = img[0:img.shape[0]//3,:]
    #cv2.imshow#("split",img_split)
    gray = cv2.cvtColor(img_split, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img_split, cv2.COLOR_BGR2HSV)
    ret, image_er = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)  # OTSU最大类间方差法
    #cv2.imshow#("img_er",image_er)
    contours, hierarchy = cv2.findContours(image_er, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 找轮廓
    # cv2.drawContours(img, contours, -1, (0, 255, 0), 3)  # 把边缘给画出来
    # #cv2.imshow#("luokuo", img)
    boundRect = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        # 一个筛选，可能需要看识别条件而定，有待优化
        if  h < 0.9*img_split.shape[0] and w/h > 2 and w>img_split.shape[1]//3:
            boundRect.append([x, y, w, h])
            # 画一个方形标注一下，看看圈的范围是否正确
            red_dil = cv2.rectangle(img_split, (x, y), (x + w, y + h), 255, 2)
            # print(x, y, w, h)
            #cv2.imshow#('red_dir', red_dil)
    if boundRect is not None:
        boundRect.sort(key=lambda x: x[2], reverse=True)
        print(boundRect)
        if boundRect is not None:
            run_led_area = [boundRect[0][0] + boundRect[0][2] // 28 * 19, boundRect[0][1], boundRect[0][2] // 7,
                            boundRect[0][3]]
            run_led_img = img_split[run_led_area[1]:run_led_area[1] + run_led_area[3],
                          run_led_area[0]:run_led_area[0] + run_led_area[2]]
            #cv2.imshow#("run_led", run_led_img)
            return run_led_img


def running_led_recog(img):
    print(img.shape[0],img.shape[1])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)
    # gray = cv2.GaussianBlur(gray,(3,3),0)
    # #cv2.imshow#("gray", gray)
    ret, image_er = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)  # OTSU最大类间方差法
    #cv2.imshow#("img_er",image_er)
    canny = cv2.Canny(gray,50,150)
    #cv2.imshow#("canny",canny)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dir = cv2.dilate(canny,kernel)
    #cv2.imshow#("dir",dir)
    contours, hierarchy = cv2.findContours(image_er, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 找轮廓
    # cv2.drawContours(img, contours, -1, (0, 255, 0), 3)  # 把边缘给画出来
    # #cv2.imshow#("luokuo", img)
    boundRect = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        # 一个筛选，可能需要看识别条件而定，有待优化
        if  h < 0.9*img.shape[0] and w/h > 1.2 and w < 0.8*img.shape[1] and w > img.shape[1]//6:
            boundRect.append([x, y, w, h])
            # 画一个方形标注一下，看看圈的范围是否正确
            red_dil = cv2.rectangle(img, (x, y), (x + w, y + h), 255, 2)
            # print(x, y, w, h)
            #cv2.imshow#('red_dir', red_dil)
    if boundRect is not None:
        boundRect.sort(key=lambda x: x[1], reverse=False)
        print(boundRect)
        led_list = []
        for i in range(2):
            image_zhong = img[boundRect[i][1]:boundRect[i][1] + boundRect[i][3],
                          boundRect[i][0] - boundRect[i][2] // 2 * 3:boundRect[i][0] - boundRect[i][2] // 2]
            #cv2.imshow#("obj{}".format(i), image_zhong)
            # #cv2.imshow#("image_yuan", image_yuan)
            hsv = cv2.cvtColor(image_zhong, cv2.COLOR_RGB2HSV)
            H, S, V = cv2.split(hsv)
            # print("HSV", H, S, V)
            v = V.ravel()[np.flatnonzero(V)]  # 亮度非零的值
            average_v = sum(v) / len(v)
            print("average_v", average_v)
            if average_v < 120:
                led_list.insert(i, '灭')
            else:
                led_list.insert(i, '亮')
        print("led_list", led_list)
        return led_list

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
        if h < img.shape[0] // 2 and w / h > 2 and w>img.shape[1]//10:
            boundRect.append([x, y, w, h])
            # 画一个方形标注一下，看看圈的范围是否正确
            red_dil = cv2.rectangle(img, (x, y), (x + w, y + h), 255, 2)
            print(x, y, w, h)
            #cvshow1000#('red_dir', red_dil)
    boundRect.sort(key=lambda x: x[1], reverse=False)
    print(boundRect)
    if boundRect is not None:
        left_area = [boundRect[0][0]-boundRect[0][2]//2, boundRect[0][1]+boundRect[0][3],boundRect[0][2],boundRect[0][3]]
        right_area = [boundRect[0][0]+boundRect[0][2]*2, boundRect[0][1]+boundRect[0][3],boundRect[0][2],boundRect[0][3]]
        left_img = img[left_area[1]:left_area[1]+left_area[3], left_area[0]:left_area[0]+left_area[2]]
        right_img = img[right_area[1]:right_area[1]+right_area[3], right_area[0]:right_area[0]+right_area[2]]
        #cv2.imshow#("left_area", left_img)
        #cv2.imshow#("right_area", right_img)
        return left_img, right_img

#这个测点因为没有阴影部分遮挡，所以采用判断黑色像素点来进行开关的上下判断
def key_recog(img):
    h,w = img.shape[0], img.shape[1]
    print(h,w)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(3,3),0)
    #cv2.imshow#("gray2", gray)
    canny = cv2.Canny(gray,150,300)
    #cv2.imshow#("cnany",canny)
    ret, image_er = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)  # OTSU最大类间方差法
    #cv2.imshow#("er_recog",image_er)
    img_up = image_er[0:h//16*9,0:w]
    #cv2.imshow#("up",img_up)
    img_down = image_er[h//16*9:h,0:w]
    len_up = len(img_up[img_up==0])
    print("len_up",len_up )
    len_down = len(img_down[img_down==0])
    print("len_down",len_down)
    #0表示开关为竖着的状态（远地开关），1表示横着的状态（就地开关）
    if len_up>len_down:
        return '上'
    else:
        return '下'

def tran(list):
    list_str = str(list).replace("[", "").replace("]", "")
    return eval(f"[{list_str}]")

def Point_twenty_six(file_name1, file_name2, file_name3):
    img1 = cv2.imread(file_name1)
    img2 = cv2.imread(file_name2)
    img3 = cv2.imread(file_name3)
    Point_list = []
    led_list, area = instruct_led_split_and_recog(img1)
    Point_list.append(led_list)
    APT_key_img = APTkey_split(img1, area)
    Point_list.append(APTkey_recogn(APT_key_img))
    # running_led_img = running_led_split(img2)
    # Point_list.append(running_led_recog(running_led_img))
    # left_img, right_img = key_split(img3)
    # Point_list.append(key_recog(left_img))
    # Point_list.append(key_recog(right_img))

    print("Point_twenty_seven", Point_list)
    return tran(Point_list)




# if __name__ == '__main__':
#     file_name = 'E:\\desktop\\images2\\27-3.JPG'
#     # img = cv2.imread(file_name)
#     # img_split = img[img.shape[0]//3*2:img.shape[0], 0:img.shape[1]//2]
#     # # APTkey_split(file_name)
#     # instruct_led_recogn(img_split)
#     # led_list, area = instruct_led_split_and_recog(file_name)
#     # img = APTkey_split(file_name, area)
#     # key = APTkey_recogn(img)
#     # print(key)
#     # img = run_led_split(file_name)
#     # run_led_recog(img)
#     left_img, right_img = key_split(file_name)
#     # right_img = cv2.imread(file_name)
#     key = key_recog(left_img)
#     print("key:",key)
#
#     # print(area)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
