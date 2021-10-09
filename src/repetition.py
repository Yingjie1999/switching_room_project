import numpy as np
import cv2

#调度号的分割和识别可以作为父类里面的一个函数
#数码管识别也可以


class Repetition(object):
    def __init__(self, img_input_path):
        img_input = cv2.imread(img_input_path)
        self.image = img_input
        #cv2.imshow#("input", self.image)

    def cvshow1000(self, name, img):
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(name, 1000, 750)
        #cv2.imshow#(name, img)

    def gain_h_and_w(self, img):
        h = img.shape[0]
        w = img.shape[1]
        return h, w

    #调度号的分割，然后需要再加一个字符的识别
    def num_split(self):
        img = self.image
        split_h, split_w = self.gain_h_and_w(img)
        image_split = img[split_h // 4:split_h // 4 * 3, split_w // 4:3 * split_w // 4]
        gray = cv2.cvtColor(image_split, cv2.COLOR_BGR2GRAY)
        image_g = cv2.GaussianBlur(gray, (3, 3), 0)
        self.cvshow1000("num_gray", gray)
        ret, image_er = cv2.threshold(image_g, 0, 255, cv2.THRESH_OTSU)  # OTSU最大类间方差法
        self.cvshow1000("er", image_er)
        contours, hierarchy = cv2.findContours(image_er, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 找轮廓
        # cv2.drawContours(image_split, contours, -1, (0, 255, 0), 3)  # 把边缘给画出来
        # self.cvshow1000("bounds",image_split)
        boundRect = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            # 一个筛选，可能需要看识别条件而定，有待优化
            if w / h > 1.5 and w < 0.9 * split_w and w > 0.1 * split_w and h < 0.9 * split_h and h > 0.1 * split_h:
                boundRect.append([x, y, w, h])
                # 画一个方形标注一下，看看圈的范围是否正确
                # red_dil = cv2.rectangle(image_split, (x, y), (x + w, y + h), 255, 2)
                # print(x, y, w, h)
                # self.cvshow1000("rectangle", red_dil)
        # 暂时通过最大值来判断
        a = np.array(boundRect)
        maxindex = a.argmax(axis=0)
        # print(maxindex)
        black = []
        black = (boundRect[maxindex[0]])
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
        # #cv2.imshow#("image_out",image_out)
        print(dispatchNum_coordinate)
        diaoduhao40 = cv2.rectangle(image_split, (dispatchNum_coordinate_x, dispatchNum_coordinate_y),
                                    (dispatchNum_coordinate_x + dispatchNum_coordinate_w,
                                     dispatchNum_coordinate_y + dispatchNum_coordinate_h), (0, 0, 255), 2)
        self.cvshow1000("num",diaoduhao40)
        return image_out




