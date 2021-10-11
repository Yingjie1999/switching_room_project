import requests
import re
from bs4 import BeautifulSoup
import os


# 打开网页,获取网页源码
def getUrl(url):

    try:
        read = requests.get(url)  #获取url
        read.raise_for_status()   #状态响应 返回200连接成功
        read.encoding = read.apparent_encoding  #从内容中分析出响应内容编码方式
        print(read.text)
        return read.text    #Http响应内容的字符串，即url对应的页面内容
    except:
        return "连接失败！"


# 正常网页获取图片地址并保存下载
def getImag(html):
    imagelist = re.findall('img src="(.*?)" ', html)

    pat = 'list/(.*?).png'
    ex = re.compile(pat)
    i = 1
    for url in imagelist:
        print('Downloding:' + url)
        # 从图片地址下载数据
        image = requests.get(url)
        #         获取英雄名（这里可以自己为文件取名就行，下面的name变量是从图片地址中提取到的英雄名）
        pat = 'list/(.*?).png'
        ex = re.compile(pat)
        if ex.search(url):
            name = ex.search(url).group(1)
        else:
            pat = 'heroes/(.*?)/hero-select'
            ex = re.compile(pat)
            if ex.search(url):
                name = ex.search(url).group(1)
            else:
                name = 'new' + str(i) + '?'
                i = i + 1
        # 在目标路径创建相应文件
        f = open('C:\\Users\\SONG\\Desktop\\getpic\\' + name + '.png', 'wb')
        # 将下载到的图片数据写入文件
        f.write(image.content)
        f.close()

    return '结束'

def gethref(html):
    soup = BeautifulSoup(html, 'html.parser')  # 文档对象
    # 查找a标签,只会查找出一个a标签

    for k in soup.find_all('a'):
        # print(k)
        print(k['href'])  # 查a标签的href值
        if k['href'][-3:] == 'jpg':
            url = 'https://slive.ploughuav.com:65455'+k['href']
            print(url)
            image = requests.get(url)
            # 在目标路径创建相应文件
            f = open('C:\\Users\\SONG\\Desktop\\getpic\\'+ k['href'][-16:], 'wb')
            # 将下载到的图片数据写入文件
            f.write(image.content)
            f.close()





# 主函数
if __name__ == '__main__':
    html_url = getUrl("https://slive.ploughuav.com:65455/Img/192.168.144.160/2021-09-29/")
    gethref(html_url)
    # getImag(html_url)
