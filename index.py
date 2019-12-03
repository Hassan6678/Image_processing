from PIL import Image

col = Image.open("wow_3.png")
gray = col.convert('L')
bw = gray.point(lambda x: 0 if x < 128 else 255, '1')
bw.save("result_bw.png")

# import numpy as np
# import cv2
# from matplotlib import pyplot as plt
# import pytesseract
# from PIL import Image

# img = cv2.imread('wow_3.png')
# #img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ret, bw_img = cv2.threshold(img,127,255,cv2.THRESH_BINARY)

# dst = cv2.medianBlur(bw_img, 3)
# #cv2.imshow('median_blur', median_blur)

# # cv2.waitKey()
# # cv2.destroyAllWindows()

# plt.subplot(131),plt.imshow(img)
# plt.subplot(132),plt.imshow(bw_img)
# plt.subplot(133),plt.imshow(dst)
# plt.show()
# import cv2

# originalImage = cv2.imread('wow_1.png')
# grayImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)

# (thresh, blackAndWhiteImage) = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY)

# dst = cv2.medianBlur(blackAndWhiteImage, 3)
# #cv2.imshow('median_blur', dst)
# cv2.imwrite('bw.png',blackAndWhiteImage)
# cv2.imwrite('gray.png',grayImage)
# plt.subplot(221),plt.imshow(originalImage)
# plt.subplot(222),plt.imshow(grayImage)
# plt.subplot(223),plt.imshow(blackAndWhiteImage)
# plt.subplot(224),plt.imshow(dst)
# plt.show()