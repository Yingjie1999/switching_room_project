

import pandas as pd

def pd_toexcel(data,file_name):
    ids = []
    serials = []
    times = []
    currents = []
    voltages = []
    frequencys = []
    ammeter_ones = []
    ammeter_twos = []
    ammeter_threes = []
    # for i in range(len(data)):
        # print(data['id'])
    ids.append(data['id'])
    serials.append(data['serial'])
    times.append(data["time"])
    currents.append(data["current"])
    voltages.append(data["voltage"])
    frequencys.append(data["frequency"])
    ammeter_ones.append(data["ammeter_one"])
    ammeter_twos.append(data["ammeter_two"])
    ammeter_threes.append(data["ammeter_three"])

    dfData = {  # 用字典设置DataFrame所需数据
        '序号': ids,
        '编号': serials,
        '时间': times,
        '电流': currents,
        '电压': voltages,
        '频率': frequencys,
        '电流表（第一行）': ammeter_ones,
        '电流表（第二行）': ammeter_twos,
        '电流表（第三行）': ammeter_threes
    }
    df = pd.DataFrame(dfData)  # 创建DataFrame
    df.to_excel(file_name, index=False)  # 存表，去除原始索引列（0,1,2...）

def ser_list_to_dic(ser_list):
    ser_key = ["serial"]
    ser_dict = dict(zip(ser_key,ser_list))
    return ser_dict

def lcd_list_to_dic(lcd_list):
    lcd_key = ["time", "current", "voltage", "frequency"]
    lcd_dict = dict(zip(lcd_key,lcd_list))
    return lcd_dict

def digit_list_to_dic(digit_list):
    digit_key = ["ammeter_one", "ammeter_two", "ammeter_three"]
    digit_dict = dict(zip(digit_key,digit_list))
    return digit_dict

