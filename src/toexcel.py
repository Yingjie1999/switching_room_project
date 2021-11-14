import xlwt
import numpy
import xlrd



def write_file(list, excel_save_path):

    flile_name = "./doc/temple2.xlsx"
    # 读取源excel
    xlsx = xlrd.open_workbook(flile_name)

    # 准备写入
    new_workbook = xlwt.Workbook(encoding='utf-8')

    table = xlsx.sheet_by_index(0)
    rows = table.nrows
    cols = table.ncols
    print("cols:", cols)
    worksheet = new_workbook.add_sheet("sheet" + str(0), cell_overwrite_ok=True)
    #写入原表格信息
    for j in range(0, len(list)):
        for i in range(0, cols):
            # print(i,j,table.cell_value(i, j)
            # worksheet.write(j, i, table.cell_value(j,i))
            if int(list[j][-1]) <= 22:
                worksheet.write(j * 4 + 1, i, table.cell_value((int(list[j][-1]) - 1) * 4 + 1, i))
                worksheet.write(j*4 +2, i, table.cell_value((int(list[j][-1]) - 1) * 4 + 2, i))
            elif int(list[j][-1]) > 22 and int (list[j][-1]) <=40:
                worksheet.write(j * 4 + 1, i, table.cell_value((int(list[j][-1])-1)*4+1, i))
                worksheet.write(j * 4 + 2, i, table.cell_value((int(list[j][-1]) - 1) * 4 + 2, i))

    #写入识别信息
    for j in range(len(list)):
        for i in range(len(list[j])):
            if int(list[j][-1]) <= 22:
                # worksheet.write(j*3+1, i, table.cell_value((int(list[j][-1])-1)*3+1, i))
                # worksheet.write(int(list[j][-1]) * 3 - 1, i, table.cell_value(int(list[j][-1]) * 3 - 2, i))
                worksheet.write(j*4+2, i+3, list[j][i]) #先行后列
            elif int(list[j][-1]) > 22 and int (list[j][-1]) <=40:
                worksheet.write(j*4+2,i+3, list[j][i])
                # worksheet.write(j*4+3, i+3, list[j][i])


    new_workbook.save(excel_save_path)

if __name__ == '__main__':
    list = [['亮', '亮', '亮', '开', '合闸'], ['亮', '亮', '亮', '远方', '预合合后', '合', '分', '关', '开', '亮', '灭', '亮', '合闸', '合闸', '分闸', '亮', '灭', '灭', '灭', '亮']]
    write_file(list)
