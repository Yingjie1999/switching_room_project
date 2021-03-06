import copy
import os
import toexcel
import shutil
import tofile
import cv2
import numpy as np
import pymssql
import code_recog
import meas_point_1
import meas_point_2
import meas_point_3
import meas_point_4
import meas_point_5
import meas_point_6
import meas_point_7
import meas_point_8
import meas_point_9
import meas_point_10
import meas_point_11
import meas_point_12
import meas_point_13
import meas_point_14
import meas_point_15
import meas_point_16
import meas_point_17
import meas_point_18
import meas_point_19
import meas_point_20
import meas_point_21
import meas_point_22
import meas_point_23
import meas_point_24
import meas_point_25
import meas_point_26
import meas_point_27
import meas_point_28
import meas_point_29
import meas_point_30
import meas_point_31
import meas_point_32
import meas_point_33
import meas_point_34
import meas_point_35
import meas_point_36
import meas_point_37
import meas_point_38
import meas_point_39
import meas_point_40
import meas_point_41
import meas_point_42

def default():
    print("----------------Output is None------------")

# 字典中不同的值对应不同的自定义函数
switcher = {
    '1': meas_point_1.Point_one,
    '2': meas_point_2.Point_two,
    '3': meas_point_3.Point_three,
    '4': meas_point_4.Point_four,
    '5': meas_point_5.Point_five,
    '6': meas_point_6.Point_six,
    '7': meas_point_7.Point_seven,
    '8': meas_point_8.Point_eight,
    '9': meas_point_9.Point_nine,
    '10':meas_point_10.Point_ten,
    '11':meas_point_11.Point_eleven,
    '12':meas_point_12.Point_twelve,
    '13':meas_point_13.Point_thirteen,
    '14':meas_point_14.Point_fourteen,
    '15':meas_point_15.Point_fifteen,
    '16':meas_point_16.Point_sixteen,
    '17':meas_point_17.Point_seventeen,
    '18':meas_point_21.Point_twenty_one,
    '19':meas_point_21.Point_twenty_one,
    '20':meas_point_20.Point_twenty,
    '21':meas_point_21.Point_twenty_one,
    '22':meas_point_22.Point_twenty_two,
    '23':meas_point_23.Point_twenty_three,
    '24':meas_point_24.Point_twenty_four,
    '25':meas_point_25.Point_twenty_five,
    '26':meas_point_26.Point_twenty_six,
    '27':meas_point_27.Point_twenty_seven,
    '28':meas_point_28.Point_twenty_eight,
    '29':meas_point_29.Point_twenty_nine,
    '30':meas_point_30.Point_thirty,
    '31':meas_point_31.Point_thirty_one,
    '32':meas_point_32.Point_thirty_two,
    '33':meas_point_33.Point_thirty_three,
    '34':meas_point_34.Point_thirty_four,
    '35':meas_point_35.Point_thirty_five,
    '36':meas_point_36.Point_thirty_six,
    '37':meas_point_37.Point_thirty_seven,
    '38':meas_point_38.Point_thirty_eight,
    '39':meas_point_39.Point_thirty_nine,
    '40':meas_point_40.Point_fourty

    # 3: get_function_3
}


def mkdir(path):
    folder = os.path.exists(path)

    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径
        print
        "---  new folder...  ---"
        print
        "---  OK  ---"

    else:
        print
        "---  There is this folder!  ---"

#
# file = "G:\\xxoo\\test"
# mkdir(file)  # 调用函数

# 根据flag的值决定执行哪一个函数，如果输入的值在字典中没有，则执行get_default函数
def modify_file_path(file_path):
    print(file_path[0:33])
    if file_path[0:33] == 'https://slive.ploughuav.com:65455':
        # print('1111')
        file_path_out = file_path.replace('https://slive.ploughuav.com:65455', 'D:/beijingImg')
    print(file_path_out)
    return file_path_out

def total_recogntion(file_path):
# if __name__ == '__main__':
    # file_path = modify_file_path(file_path)

    file_path = 'C:/Users/SONG/Desktop/image6/'
    file_path2 = file_path + 'result/'
    # file_path3 = file_path2.replace('/', '\\')
    # print(file_path3)
    # file_path3 = "C:\\Users\\SONG\\Desktop\\image6\\result"
    mkdir(file_path2)
    excel_save_path = file_path + 'result/test.xls'
    dstpath = file_path + 'result/'
    file_list = code_recog.getfileorder(file_path)
    Total_list = []
    showpic_num = 0

    for i in range(len(file_list)):
        image_path = file_path + file_list[i]
        code_info = code_recog.ocr_qrcode_zxing(image_path)
        print(code_info)
        if code_info.parsed is not '':
            # 根据flag的值决定执行哪一个函数，如果输入的值在字典中没有，则执行get_default函数
            print("识别到二维码，为 "+ code_info.parsed)
            output = switcher[code_info.parsed](image_path, code_info)
            output.append(code_info.parsed)
            Total_list.append(output)
            showpic_num += 1
        else:
            Total_list.append([' '])
    # file_list_len = len(file_list)
    print("show_pic_num:", showpic_num)
    print(Total_list)
    output_list = copy.deepcopy(Total_list)
    pop_num = 0
    # 判断两张相同测点照片哪张涵盖信息多
    for i in range(len(Total_list)-1):
        if Total_list[i][-1] == Total_list[i+1][-1] and Total_list[i][0] != '1' and Total_list[i][0] != '2' and Total_list[i][0] != '3':
            sum1 = 0 ;sum2 = 0
            for j in Total_list[i]:
                if j and j is not ' ':
                    sum1 = sum1+1
            len1 = len(Total_list[i]) - sum1
            # print("len1",len1)
            for j in Total_list[i+1]:
                if j and j is not ' ':
                    sum2 = sum2+1
            len2 = len(Total_list[i+1]) - sum2
            # print("len2:",len2)
            if len1 <= len2:
                output_list.pop(i+1-pop_num)
                pop_num += 1
                # print(pop_num)
                tofile.copy_and_move(file_path + file_list[i], dstpath, Total_list[i][-1]+'-1.jpg')
            else:
                output_list.pop(i-pop_num)
                pop_num += 1
                # print(pop_num)
                tofile.copy_and_move(file_path + file_list[i+1], dstpath, Total_list[i+1][-1]+'-1.jpg')

        elif Total_list[i][0] == '1':
            output_list[i-pop_num].pop(0)

        elif Total_list[i][0] == '2':
            # print('222222')
            output_list[i-1-pop_num].pop(-1)
            output_list[i-pop_num].pop(0)
            output_list[i-1-pop_num] += output_list[i-pop_num]
            output_list.pop(i-pop_num)
            pop_num += 1
            print(pop_num)
            print(output_list)
            if Total_list[i+1][0] == '3':
                # print('33332')
                output_list[i - pop_num].pop(-1)
                output_list[i + 1 - pop_num].pop(0)
                output_list[i - pop_num] += output_list[i + 1 - pop_num]
                output_list.pop(i +1 - pop_num)
                pop_num += 1
                print(pop_num)
        elif Total_list[i][-1] == ' ':
            output_list.pop(i-pop_num)
            pop_num += 1
        else:
            tofile.copy_and_move(file_path + file_list[i], dstpath, Total_list[i][-1]+'-1.jpg')


    output_list.sort(key=lambda x: int(x[-1]), reverse=False)
    # for i in range(len(output_list)):
    #     output_list[i].pop(-1)
    print("output_list:", output_list)
    toexcel.write_file(output_list, excel_save_path)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

# if  __name__ == '__main__':
#     file_path = 'https://slive.ploughuav.com:65455/Img/192.168.144.160/copy/'
#     modify_file_path(file_path)
