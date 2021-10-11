import os
import logging
from PIL import Image
import zxing  # 导入解析包
import random
import cv2
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)  # 记录数据

if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

DEBUG = (logging.getLevelName(logger.getEffectiveLevel()) == 'DEBUG')  # 记录调式过程

# 在当前目录生成临时文件，规避路径问题
def ocr_qrcode_zxing(filename):
    Img = Image.open(filename)
    #对输入的图像进行分割处理，取中间1/2
    h = Img.height
    w = Img.width
    img = Img.crop([w//4,0,w//4*3,h])
    # plt.imshow(img)
    # plt.show()
    ran = int(random.random() * 100000)  # 设置随机数据的大小
    img.save('%s%s.jpg' % (os.path.basename(filename).split('.')[0], ran))
    zx = zxing.BarCodeReader()  # 调用zxing二维码读取包
    data = ''
    zxdata = zx.decode('%s%s.jpg' % (os.path.basename(filename).split('.')[0], ran))  # 图片解码

    # 删除临时文件
    os.remove('%s%s.jpg' % (os.path.basename(filename).split('.')[0], ran))
    if zxdata:
        logger.debug(u'zxing识别二维码:%s,内容: %s' % (filename, zxdata))
        data = zxdata
    else:
        logger.error(u'识别zxing二维码出错:%s' % (filename))
        img.save('%s-zxing.jpg' % filename)
    return data  # 返回记录的内容


def getfileorder(file_path):
    files = os.listdir(file_path)
    files.sort(key=lambda x: int(x.split('.')[0]))
    print(files)
    return files


#目前25二维码识别不了
if __name__ == "__main__" :
    # file_path = 'C:\\Users\\SONG\\Desktop\\image5\\'
    # file_list = getfileorder(file_path)
    # print(len(getfileorder(file_path)))

    file_name = 'C:\\Users\\SONG\\Desktop\\image5\\22_15_59_950.jpg'
    # zxing二维码识别，界面中只能有一个二维码
    ltext = ocr_qrcode_zxing(file_name)  # 将图片文件里的信息转码放到ltext里面
    # logger.info(u'[%s]Zxing二维码识别:[%s]!!!' % (filename, ltext))  # 记录文本信息
    print(ltext)  # 打印出二维码名字

