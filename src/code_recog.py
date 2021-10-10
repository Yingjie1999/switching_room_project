import zxing
import sys
reader = zxing.BarCodeReader()
barcode = reader.decode("C:\\Users\\SONG\\Desktop\\images3\\31.jpg")
print(barcode.parsed)