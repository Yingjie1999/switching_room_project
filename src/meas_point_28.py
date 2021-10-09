import numpy
import cv2



def tran(list):
    list_str = str(list).replace("[", "").replace("]", "")
    return eval(f"[{list_str}]")

def Point_twenty_eight(file_name1, file_name2):
    img1 = cv2.imread(file_name1)
    img2 = cv2.imread(file_name2)

    Point_list = []
    Point_list.append([' '])
    print("Point_twenty_eight", Point_list)
    return tran(Point_list)