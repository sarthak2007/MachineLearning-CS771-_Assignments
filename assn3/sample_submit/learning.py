import cv2
import numpy as np

filename = "../train/BMVP.png"
image = cv2.imread(filename)
hsv_image = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)

# kernel = np.ones((2,2), np.uint8)
# img_erosion = cv2.erode(hsv_image, kernel, iterations=3) 
# img_dilation = cv2.dilate(hsv_image, kernel, iterations=1) 

for i in range(150):
	for j in range(600):
		if hsv_image[i][j][2] == 204:
			hsv_image[i][j] = hsv_image[0][0]

# ret, thresh = cv2.threshold(hsv_image, 10, 255, cv2.THRESH_OTSU)
gray_image = hsv_image[:,:,2]
# # converting gray to black
# for i in range(150):
# 	for j in range(600):
# 		if gray_image[i][j] != 255:
# 			gray_image[i][j]=0

flag, start, end = 0, -1, 0
segmented = []
for i in range(600):
	flag = 0
	for j in range(150):
		# print(gray_image[j][i])
		# if gray_image[j][i] == 204:
		# 	gray_image[j][i] = 255

		if gray_image[j][i] != 255:
			if start == -1:
				start = i
			flag = 1
			break
	if start != -1 and flag == 0:
		# print(start, end)
		end = i - 1
		segmented.append(gray_image[:,start:end])
		start = -1


cnt = 0

# for arr in segmented:
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