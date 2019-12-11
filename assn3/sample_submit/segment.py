import cv2
import numpy as np
import os
# from skimage import measure

# filepath = os.listdir("../train")
filename = "../train/XTOV.png"
# for filename in filepath:
image = cv2.imread(filename)
hsv_image = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)

ret, gray_image = cv2.threshold(hsv_image[:,:,2], 203, 255, cv2.THRESH_BINARY)

flag, start, end = 0, -1, 0
segmented = []

# labels = measure.label(gray_image, neighbors=8, background=0)
# charCandidates = np.zeros(gray_image, dtype="uint8")
# edged=cv2.Canny(gray_image,30,200)
# cv2.imshow('canny edges',edged)
# contours, hierarchy=cv2.findContours(edged,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
# cv2.imshow('canny edges after contouring', edged)
# print(contours)
# cv2.drawContours(gray_image, contours, -1, (0, 255, 0), 3) 
  
# cv2.imshow('Contours', gray_image) 
# # print(labels.shape)
# gray_image = np.array(gray_image)
# val = np.where(gray_image==0)
# print(val.shape)
for i in range(600):
	flag = 0
	for j in range(150):
		if gray_image[j][i] != 255:
			if start == -1:
				start = i
			flag = 1
			break
	if start != -1 and flag == 0:
		end = i - 1
		if end - start < 40:
			start = -1
			continue
		resize = gray_image[5:145, start:end]
		# print(start, end)
		length = end - start
		size = (140 - length)//2
		temp = np.ones((140, size))
		resize = np.hstack((resize,temp*255))
		if length % 2 == 1:
			temp = np.hstack((temp, np.ones((140, 1))))
		resize = np.hstack((temp*255, resize))

		segmented.append(resize)
		start = -1

# cnt = 0
# for arr in segmented:
# 	print(arr.shape)
# 	cv2.imshow('i'+str(cnt), arr)
# 	cnt+=1
# print(segmented[0])


# for i in gray_image:
# 	print(i[290])

	# print(i)
# print (hsv_image[49])
# print(gray_image.shape)
# cv2.imshow('image', gray_image)
# cv2.imshow('hsv_image', gray_image[:,290:300])
# cv2.imshow('erosion', img_erosion)
# cv2.imshow('dilation', img_dilation)
cv2.waitKey(0)