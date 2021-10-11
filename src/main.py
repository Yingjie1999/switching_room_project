import cv2
import numpy as np
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
    # 2: get_function_2,
    # 3: get_function_3
}
# 根据flag的值决定执行哪一个函数，如果输入的值在字典中没有，则执行get_default函数

if __name__ == '__main__':
    file_path = 'C:\\Users\\SONG\\Desktop\\image6\\'
    file_list = code_recog.getfileorder(file_path)

    for i in range(len(file_list)):
        image_path = file_path + file_list[i]
        code_info = code_recog.ocr_qrcode_zxing(image_path)
        print(code_info)
        if code_info.parsed is not '':
            # 根据flag的值决定执行哪一个函数，如果输入的值在字典中没有，则执行get_default函数
            print("识别到二维码，为 "+ code_info.parsed)
            output = switcher[code_info.parsed](image_path, code_info)




    cv2.waitKey(0)
    cv2.destroyAllWindows()