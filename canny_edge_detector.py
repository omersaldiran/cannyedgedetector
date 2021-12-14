import numpy as np
from cv2 import cv2 as cv

# Steps of the Canny edge detection algorithm
# 1- Noise reduction
# 2- Gradient calculation
# 3- Non-maximum suppression
# 4- Double threshold
# 5- Hysteresis edge tracking


# Reading input images
santorini_image = cv.imread("santorini.jpg")

street_image = cv.imread("street.jpg")

village_image = cv.imread("village.jpg")

# Show original images
cv.imshow("Original Santorini Image", santorini_image)

cv.imshow("Original Street Image", street_image)

cv.imshow("Original Village Image", village_image)

# First convert all images to grayscale
gray_santorini_image = cv.cvtColor(santorini_image, cv.COLOR_BGRA2GRAY)

gray_street_image = cv.cvtColor(street_image, cv.COLOR_BGRA2GRAY)

gray_village_image = cv.cvtColor(village_image, cv.COLOR_BGRA2GRAY)

# Show grayscale of original images
cv.imshow("Grayscale Santorini Image", gray_santorini_image)

cv.imshow("Grayscale Street Image", gray_street_image)

cv.imshow("Grayscale Village Image", gray_village_image)

# Applying Gaussian kernel to smooth the images for Noise Reduction with (5,5) kernel
blur_santorini = cv.GaussianBlur(gray_santorini_image, (5,5), 0)

blur_street = cv.GaussianBlur(gray_street_image, (5,5), 0)

blur_village = cv.GaussianBlur(gray_village_image, (5,5), 0)

cv.imshow("Blurred Santorini", blur_santorini)

cv.imshow("Blurred Street", blur_street)

cv.imshow("Blurred Village", blur_village)

# Apply Sobel to calculate the Gradient of blurred images
gx = cv.Sobel(np.float32(blur_santorini), cv.CV_64F, 1, 0, 3)
gy = cv.Sobel(np.float32(blur_santorini), cv.CV_64F, 0, 1, 3)

# Conversion of Cartesian coordinates to polar
size, ang = cv.cartToPolar(gx, gy, angleInDegrees = True)


# Non-maximum suppression step
def non_max_supp(img, threshold1 = None, threshold2 = None):

    # setting the minimum and maximum thresholds  for double thresholding
    max_size = np.max(size)
    if not threshold1:threshold1 = max_size * 0.1
    if not threshold2:threshold2 = max_size * 0.5

    # getting the dimensions of the input image
    height, width = img.shape

    # Looping through every pixel of the grayscale  image
    for i_x in range(width):
        for i_y in range(height):

            grad_ang = ang[i_y, i_x]
            grad_ang = abs(grad_ang-180) if abs(grad_ang)>180 else abs(grad_ang)

            # selecting the neighbours of the target pixel
            # according to the gradient direction
            # In the x axis direction
            if grad_ang<= 22.5:
                neighb_1_x, neighb_1_y = i_x-1, i_y
                neighb_2_x, neighb_2_y = i_x + 1, i_y

            # top right (diagnol-1) direction
            elif grad_ang>22.5 and grad_ang<=(22.5 + 45):
                neighb_1_x, neighb_1_y = i_x-1, i_y-1
                neighb_2_x, neighb_2_y = i_x + 1, i_y + 1

            # In y-axis direction
            elif grad_ang>(22.5 + 45) and grad_ang<=(22.5 + 90):
                neighb_1_x, neighb_1_y = i_x, i_y-1
                neighb_2_x, neighb_2_y = i_x, i_y + 1

            # top left (diagnol-2) direction
            elif grad_ang>(22.5 + 90) and grad_ang<=(22.5 + 135):
                neighb_1_x, neighb_1_y = i_x-1, i_y + 1
                neighb_2_x, neighb_2_y = i_x + 1, i_y-1

            # Now it restarts the cycle
            elif grad_ang>(22.5 + 135) and grad_ang<=(22.5 + 180):
                neighb_1_x, neighb_1_y = i_x-1, i_y
                neighb_2_x, neighb_2_y = i_x + 1, i_y

            # Non-maximum suppression step
            if width>neighb_1_x>= 0 and height>neighb_1_y>= 0:
                if size[i_y, i_x]<size[neighb_1_y, neighb_1_x]:
                    size[i_y, i_x]= 0
                    continue

            if width>neighb_2_x>= 0 and height>neighb_2_y>= 0:
                if size[i_y, i_x]<size[neighb_2_y, neighb_2_x]:
                    size[i_y, i_x]= 0

    return size

# Edge Tracking by Hysteresis
def hysteresis(img, weak, strong=255):
    M, N = img.shape
    for i in range(1, M-1):
        for j in range(1, N-1):
            if (img[i,j] == weak):
                try:
                    if ((img[i+1, j-1] == strong) or (img[i+1, j] == strong) or (img[i+1, j+1] == strong)
                        or (img[i, j-1] == strong) or (img[i, j+1] == strong)
                        or (img[i-1, j-1] == strong) or (img[i-1, j] == strong) or (img[i-1, j+1] == strong)):
                        img[i, j] = strong
                    else:
                        img[i, j] = 0
                except IndexError as e:
                    pass
    return img

nms = non_max_supp(gray_santorini_image)

hyst_img = hysteresis(nms, 100)

cv.imshow("NMS Santorini", nms)

cv.imshow("Hysteresis", hyst_img)


cv.waitKey(0)
