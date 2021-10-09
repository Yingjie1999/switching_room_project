

from paddleocr import PaddleOCR, draw_ocr
import numpy as np
import time


time_start=time.time()
# 模型路径下必须含有model和params文件，如果没有，现在可以自动下载了，不过是最简单的模型
# use_gpu 如果paddle是GPU版本请设置为 True
ocr = PaddleOCR(use_angle_cls=True, use_gpu=False)

img_path = './images/AH222.jpg'  # 这个是自己的图片，自行放置在代码目录下修改名称

result = ocr.ocr(img_path, cls=True)
for line in result:
    print(line)
# 显示结果
print(np.shape(line))
from PIL import Image

image = Image.open(img_path).convert('RGB')
boxes = [line[0] for line in result]
print('boxes:',boxes)
txts = [line[1][0] for line in result]
print('txt:',txts)
print(type(txts))
scores = [line[1][1] for line in result]
im_show = draw_ocr(image, boxes, txts, scores)
im_show = Image.fromarray(im_show)
im_show.save('./result/result9.jpg')  # 结果图片保存在代码同级文件夹中。

time_end=time.time()
print('time cost',time_end-time_start,'s')
