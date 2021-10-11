import cv2
import numpy as np
# import pytesseract
import toexcel
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

import xlwt

def tran(list):
    list_str = str(list).replace("[", "").replace("]", "")
    print(eval(f"[{list_str}]"))



if __name__ == '__main__':
    Total_list = []
    file_name = 'C:\\Users\\SONG\\desktop\\images3\\'
    excel_save_path = 'D:\\test.xls'
    file_name1_1 = file_name + '1-1.JPG'
    file_name1_2 = file_name + '1-2.JPG'
    file_name2_1 = file_name + '2-1.JPG'
    file_name2_2 = file_name + '2-2.JPG'
    file_name3_1 = file_name + '3-1.JPG'
    file_name3_2 = file_name + '3-2.JPG'
    file_name4_1 = file_name + '4-1.JPG'
    file_name4_2 = file_name + '4-2.JPG'
    file_name5_1 = file_name + '5-1.JPG'
    file_name5_2 = file_name + '5-2.JPG'
    file_name6_1 = file_name + '6-1.JPG'
    file_name6_2 = file_name + '6-2.JPG'
    file_name7_1 = file_name + '7-1.JPG'
    file_name7_2 = file_name + '7-2.JPG'
    file_name8_1 = file_name + '8-1.JPG'
    file_name8_2 = file_name + '8-2.JPG'
    file_name9_1 = file_name + '9-1.JPG'
    file_name9_2 = file_name + '9-2.JPG'
    file_name10_1 = file_name + '10-1.JPG'
    file_name10_2 = file_name + '10-2.JPG'
    file_name11_1 = file_name + '11-1.JPG'
    file_name11_2 = file_name + '11-2.JPG'
    file_name12_1 = file_name + '12-1.JPG'
    file_name12_2 = file_name + '12-2.JPG'
    file_name13_1 = file_name + '13-1.JPG'
    file_name13_2 = file_name + '13-2.JPG'
    file_name14_1 = file_name + '14-1.JPG'
    file_name14_2 = file_name + '14-2.JPG'
    file_name15_1 = file_name + '15-1.JPG'
    file_name15_2 = file_name + '15-2.JPG'
    file_name16_1 = file_name + '16-1.JPG'
    file_name16_2 = file_name + '16-2.JPG'
    file_name17_1 = file_name + '17-1.JPG'
    file_name17_2 = file_name + '17-2.JPG'
    file_name18_1 = file_name + '18-1.JPG'
    file_name18_2 = file_name + '18-2.JPG'
    file_name19_1 = file_name + '19-1.JPG'
    file_name19_2 = file_name + '19-2.JPG'
    file_name20_1 = file_name + '20-1.JPG'
    file_name20_2 = file_name + '20-2.JPG'
    file_name21_1 = file_name + '21-1.JPG'
    file_name21_2 = file_name + '21-2.JPG'
    file_name22_1 = file_name + '22-1.JPG'
    file_name22_2 = file_name + '22-2.JPG'
    file_name23_1 = file_name + '23-1.JPG'
    file_name23_2 = file_name + '23-2.JPG'
    file_name24_1 = file_name + '24-1.JPG'
    file_name24_2 = file_name + '24-2.JPG'
    file_name25_1 = file_name + '25-1.JPG'
    file_name25_2 = file_name + '25-2.JPG'
    file_name25_3 = file_name + '25-3.JPG'
    file_name26_1 = file_name + '456.jpg'
    file_name26_2 = file_name + '26-2.JPG'
    file_name26_3 = file_name + '26-3.JPG'
    file_name27_1 = file_name + '123.jpg'
    file_name27_2 = file_name + '27-2.JPG'
    file_name28_1 = file_name + '28-1.JPG'
    file_name28_2 = file_name + '28-2.JPG'
    file_name29_1 = file_name + '29-1.JPG'
    file_name29_2 = file_name + '29-2.JPG'
    file_name30_1 = file_name + '30-1.JPG'
    file_name30_2 = file_name + '30-2.JPG'
    file_name31_1 = file_name + '31-1.JPG'
    file_name31_2 = file_name + '31-2.JPG'
    file_name32_1 = file_name + '32-1.JPG'
    file_name32_2 = file_name + '32-2.JPG'
    file_name33_1 = file_name + '33-1.JPG'
    file_name33_2 = file_name + '33-2.JPG'
    file_name34_1 = file_name + '34-1.JPG'
    file_name34_2 = file_name + '34-2.JPG'
    file_name35_1 = file_name + '35-1.JPG'
    file_name35_2 = file_name + '35-2.JPG'
    file_name36_1 = file_name + '36-1.JPG'
    file_name36_2 = file_name + '36-2.JPG'
    file_name37_1 = file_name + '37-1.JPG'
    file_name37_2 = file_name + '37-2.JPG'
    file_name38_1 = file_name + '38-1.JPG'
    file_name38_2 = file_name + '38-2.JPG'
    file_name39_1 = file_name + '39-1.JPG'
    file_name39_2 = file_name + '39-2.JPG'
    file_name40_1 = file_name + '40-1.JPG'
    file_name40_2 = file_name + '40-2.JPG'
    file_name41_1 = file_name + '41-1.JPG'
    file_name41_2 = file_name + '41-2.JPG'
    file_name42_1 = file_name + '42-1.JPG'
    # Total_list.append(meas_point_1.Point_one(file_name1_1, file_name1_2))
    # Total_list.append(meas_point_2.Point_two(file_name2_1, file_name2_2))
    # Total_list.append(meas_point_3.Point_three(file_name3_1,file_name3_2))
    # Total_list.append(meas_point_4.Point_four(file_name4_1, file_name4_2))
    # Total_list.append(meas_point_5.Point_five(file_name5_1, file_name5_2))
    # Total_list.append(meas_point_6.Point_six(file_name6_1, file_name6_2))
    # Total_list.append(meas_point_7.Point_seven(file_name7_1,file_name7_2))
    # Total_list.append(meas_point_8.Point_eight(file_name8_1,file_name8_2))
    # Total_list.append(meas_point_9.Point_nine(file_name9_1,file_name9_2))
    # Total_list.append(meas_point_10.Point_ten(file_name10_1, file_name10_2))
    # Total_list.append(meas_point_11.Point_eleven(file_name11_1, file_name11_2))
    # Total_list.append(meas_point_12.Point_twelve(file_name12_1, file_name12_2))
    # Total_list.append(meas_point_13.Point_thirteen(file_name13_1, file_name13_2))
    # Total_list.append(meas_point_14.Point_fourteen(file_name14_1, file_name14_2))
    # Total_list.append(meas_point_15.Point_fifteen(file_name15_1, file_name15_2))
    # Total_list.append(meas_point_16.Point_sixteen(file_name16_1, file_name16_2))
    # Total_list.append(meas_point_17.Point_seventeen(file_name17_1, file_name17_2))
    # Total_list.append(meas_point_18.Point_eighteen(file_name18_1, file_name18_2))
    # Total_list.append(meas_point_19.Point_nineteen(file_name19_1, file_name19_2))
    # Total_list.append(meas_point_20.Point_twenty(file_name20_1, file_name20_2))
    # Total_list.append(meas_point_21.Point_twenty_one(file_name21_1, file_name21_2))
    # Total_list.append(meas_point_22.Point_twenty_two(file_name22_1, file_name22_2))
    # Total_list.append(meas_point_23.Point_twenty_three(file_name23_1, file_name23_2))
    # Total_list.append(meas_point_24.Point_twenty_four(file_name24_1, file_name24_2))
    # Total_list.append(meas_point_25.Point_twenty_five(file_name25_1, file_name25_2, file_name25_3))
    Total_list.append(meas_point_26.Point_twenty_six(file_name26_1, file_name26_2, file_name26_3))
    # Total_list.append(meas_point_27.Point_twenty_seven(file_name27_1, file_name27_2))
    # Total_list.append(meas_point_28.Point_twenty_eight(file_name28_1,file_name28_2))
    # Total_list.append(meas_point_29.Point_twenty_nine(file_name29_1,file_name29_2))
    # Total_list.append(meas_point_30.Point_thirty(file_name30_1))
    # Total_list.append(meas_point_31.Point_thirty_one(file_name31_1,file_name31_2))
    # Total_list.append(meas_point_32.Point_thirty_two(file_name32_1,file_name32_2))
    # Total_list.append(meas_point_33.Point_thirty_three(file_name33_1, file_name33_2))
    # Total_list.append(meas_point_34.Point_thirty_four(file_name34_1,file_name34_2))
    # Total_list.append(meas_point_35.Point_thirty_five(file_name35_1, file_name35_2))
    # Total_list.append(meas_point_36.Point_thirty_six(file_name36_1, file_name36_2))
    # Total_list.append(meas_point_37.Point_thirty_seven(file_name37_1, file_name37_2))
    # Total_list.append(meas_point_38.Point_thirty_eight(file_name38_1, file_name38_2))
    # Total_list.append(meas_point_39.Point_thirty_nine(file_name39_1, file_name39_2))
    # Total_list.append(meas_point_40.Point_fourty(file_name40_1, file_name40_2))
    # Total_list.append(meas_point_41.Point_fourty_one(file_name41_1,file_name41_2))
    # Total_list.append(meas_point_42.Point_fourty_two(file_name42_1))

    print("Total:",Total_list)
    # toexcel.write_file(Total_list, excel_save_path)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
