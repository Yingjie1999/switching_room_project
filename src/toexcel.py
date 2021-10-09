import xlwt
import numpy
import xlrd



def write_file(list, excel_save_path):

    flile_name = "temple.xlsx"
    # 读取源excel
    xlsx = xlrd.open_workbook(flile_name)

    # 准备写入
    new_workbook = xlwt.Workbook(encoding='utf-8')

    table = xlsx.sheet_by_index(0)
    rows = table.nrows
    cols = table.ncols
    worksheet = new_workbook.add_sheet("sheet" + str(0), cell_overwrite_ok=True)
    for i in range(0, rows):
        for j in range(0, cols):
            # print(i,j,table.cell_value(i, j))
            worksheet.write(i, j, table.cell_value(i, j))

    for j in range(len(list)):
        for i in range(len(list[j])):
            worksheet.write((j+1)*4-1, i+3, list[j][i])  # 第0行第0列


    new_workbook.save(excel_save_path)

if __name__ == '__main__':
    list = [['亮', '亮', '亮', '开', '合闸'], ['亮', '亮', '亮', '远方', '预合合后', '合', '分', '关', '开', '亮', '灭', '亮', '合闸', '合闸', '分闸', '亮', '灭', '灭', '灭', '亮']]
    write_file(list)
