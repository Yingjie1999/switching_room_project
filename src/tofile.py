import os
import shutil

def copy_and_move(srcfile, dstpath, newname):                       # 复制函数
    if not os.path.isfile(srcfile):
        print ("%s not exist!"%(srcfile))
    else:
        fpath,fname=os.path.split(srcfile)             # 分离文件名和路径
        print(fpath)
        print(fname)
        if not os.path.exists(dstpath):
            os.makedirs(dstpath)                       # 创建路径
        shutil.copy(srcfile, dstpath + newname)          # 复制文件
        print ("copy %s -> %s"%(srcfile, dstpath + newname))



if __name__ == '__main__':

    srcfile = 'C:\\Users\\SONG\\Desktop\\image6\\22_14_45_520.jpg'
    dstfile = 'C:/Users/SONG/Desktop/CIAIC/code/switching_room_Project/src2/22_14_45_52071943.jpg'
    copy_and_move(srcfile, dstfile)
