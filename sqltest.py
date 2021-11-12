import pymssql #引入pymssql模块
import xlwt
import xlrd



def conn():
    connect = pymssql.connect(host='127.0.0.1', port='1433',  user='sa', password='123456789',database='Project') #服务器名,账户,密码,数据库名
    if connect:
        print("连接成功!")
    return connect

# def writetosql():
def getlist_table(file_name):
    list_table = []
    # 读取源excel
    xlsx = xlrd.open_workbook(file_name)
    table = xlsx.sheet_by_index(0)
    rows = table.nrows
    cols = table.ncols
    for i in range(0,cols):
        list_table.append(table.cell_value(1, i))
    print(list_table)
    return list_table

if __name__ == '__main__':
    file_name = './test.xls'
    getlist_table(file_name)
    conn = conn()
    cursor = conn.cursor()
    list_table = []

    sql_create = "create table Point_one("

    cursor.execute(sql_create)
    # sql = "create table C_test( id varchar(20) sex varchar(20)"
    # sql = "insert into C_test (id, name, sex)values(1002, 'zhangsi', 'nv')"
    # cursor.execute("""
    # IF OBJECT_ID('number','U') IS NOT NULL
    #     DROP TABLE number
    # CREATE TABLE number(
    #     ID	INT	NOT NULL,
    #     Url	VARCHAR(30)	NOT NULL
    # )
    # """)  # 执行sql语句
    # ("""
    # IF OBJECT_ID('number','U') IS NOT NULL
    #     DROP TABLE number
    # CREATE TABLE number(
    #     ID	INT	NOT NULL,
    #     Url	VARCHAR(30)	NOT NULL
    # )
    # """)
    conn.commit()  # 提交
    cursor.close()
    conn.close()

