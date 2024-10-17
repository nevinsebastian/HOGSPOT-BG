import cv2
import numpy as np
from PIL import Image

img = cv2.imread('SIGN.JPG')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

largest_contour = max(contours, key=cv2.contourArea)

mask = np.zeros_like(gray)
cv2.drawContours(mask, [largest_contour], -1, 255, -1)

result = cv2.bitwise_and(img, img, mask=mask)

cv2.imwrite('signature_no_background.jpg', result)