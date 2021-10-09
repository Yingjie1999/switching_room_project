import cv2
import numpy as np


# 
# def #cvshow1000#(name, img):
#      cv2.namedWindow(name, cv2.WINDOW_NORMAL)
#      cv2.resizeWindow(name, 1000, 750)
#      #cv2.imshow#(name, img)

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
        #cv2.imshow#("left_img", left_img)
        #cv2.imshow#("right_img", right_img)
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
        #cv2.imshow#("up", img_up_img)
        #cv2.imshow#("down", img_down_img)
        return img_up_img, img_down_img




def key_recog(img):
    h,w = img.shape[0], img.shape[1]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(3,3),0)
    #cv2.imshow#("gray2", gray)
    canny = cv2.Canny(gray,150,300)
    #cv2.imshow#("canny",canny)
    ret, image_er = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)  # OTSU最大类间方差法
    #cv2.imshow#("er_recog",image_er)
    th2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 5)
    #cv2.imshow#("th2",th2)
    img_dis = []
    lines = []
    key_list = []
    k_sum = []
    # #cv2.imshow#("dis",img_dis)
    for i in range(5):
        img_dis.append(th2[ h//4:h//4*3, i*w//5:(i+1)*w//5 ])
        lines = cv2.HoughLinesP(img_dis[i], 1, np.pi / 180, threshold=20, minLineLength=130)
        # #cv2.imshow#("dis",img_dis)
        #cv2.imshow#('obj{}'.format(i),img_dis[i])
        k = []
        if lines is not None:
            for x1, y1, x2, y2 in lines[:, 0]:
                cv2.line(img, (x1 + i * w // 5, y1 + h // 4), (x2 + i * w // 5, y2 + h // 4), (0, 0, 255), 2)
                k.append((y1 + y2) / 2 + h // 4)
                print("line", lines)
            #cv2.imshow#("line",img)
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
    #cv2.imshow#("gray2", gray)
    canny = cv2.Canny(gray, 150, 300)
    #cv2.imshow#("cnany", canny)
    ret, image_er = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)  # OTSU最大类间方差法
    #cv2.imshow#("er_recog", image_er)
    th2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 5)
    #cv2.imshow#("th", th2)
    img_dis = []
    lines = []
    key_list = []
    k_sum = []
    # #cv2.imshow#("dis",img_dis)
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

def tran(list):
    list_str = str(list).replace("[", "").replace("]", "")
    return eval(f"[{list_str}]")


def Point_twenty_five(file_name1, file_name2, file_name3):
    img1 = cv2.imread(file_name1)
    img2 = cv2.imread(file_name2)
    img3 = cv2.imread(file_name3)
    Point_list = []
    Point_list.append(knife_switch_split_and_recog(img2))
    img_up, img_down = key_split(img3)
    Point_list.append(key_recog(img_up))
    Point_list.append(key_recog(img_down))

    print("point_twenty_six:", Point_list)
    return tran(Point_list)


#
# if __name__ == '__main__':
#     file_name = 'E:\\desktop\\images2\\26-3.JPG'
#     # img = cv2.imread(file_name)
#     # key_recog(img)
#     # knife_switch_split2(file_name)
#     # print(key)
#     img_up, img_down = key_split(file_name)
#     key_up_list = key_recog(img_down)
#     # key_down_list = key_recog(img_down)
#
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
