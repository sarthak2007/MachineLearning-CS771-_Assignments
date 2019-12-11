import cv2
import numpy as np

def hor_sym(length, resize):
	size = (120 - length)//2
	temp = np.ones((150, size))
	resize = np.hstack((resize,temp*255))
	if length % 2 == 1:
		temp = np.hstack((temp, np.ones((150, 1))))
	resize = np.hstack((temp*255, resize))
	return resize

def ver_sym(length, resize):
	size = (120 - length)//2
	temp = np.ones((size, 120))
	resize = np.vstack((resize,temp*255))
	if length % 2 == 1:
		temp = np.vstack((temp, np.ones((1, 120))))
	resize = np.vstack((temp*255, resize))
	return resize

filename = "../train/AISL.png"

image = cv2.imread(filename)
hsv_image = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)

_, gray_image = cv2.threshold(hsv_image[:,:,2], 203, 255, cv2.THRESH_BINARY)


val = np.unique(np.where(gray_image==0)[1])
l = len(val)
flag, start, end = 0, val[0], -1
segmented = []
# print(val)
# result = list(zip(val, np.diff(val)))
# temp = np.where(result != 0)
# print(temp)
for i in range(l):
	if i == l-1:
		end = val[i]
	else:
		if val[i+1]-val[i] != 1:
			end = val[i]
	if end != -1:
		if end - start >= 20:
			# print(start, end)
			resize = gray_image[:, start:end]
			resize = hor_sym(end-start, resize)

			hor_zeros = np.unique(np.where(resize==0)[0])
			resize = resize[hor_zeros[0]:hor_zeros[-1]+1, :]			
			resize = ver_sym(hor_zeros[-1]-hor_zeros[0]+1, resize)

			segmented.append(resize)
		if i < l-1:
			start = val[i+1]
		end = -1

cnt = 0
for arr in segmented:
	print(arr.shape)
	cv2.imshow('i'+str(cnt), arr)
	cnt+=1
# print(segmented[0])


cv2.imshow('image', gray_image)
# cv2.imshow('hsv_image', gray_image[:,290:300])
cv2.waitKey(0)