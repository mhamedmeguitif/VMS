import cv2 #OpenCv
from matplotlib import  pyplot as plt
import numpy as np
import imutils
import easyocr
import csv
from datetime import datetime




img = cv2.imread('image1.jpg')
gray = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
plt.imshow(cv2.cvtColor(gray, cv2.COLOR_RGB2BGR ))
#bfilter = cv2.bilateralFilter(src, d, sigmaColor, sigmaSpace)
bfilter = cv2.bilateralFilter(gray, 11,17,17) #Noise detection 
edged = cv2.Canny(bfilter, 30,200) # use Canny Algorithme 
plt.imshow(cv2.cvtColor(edged, cv2.COLOR_RGB2BGR ))

KeyPoints = cv2.findContours(edged.copy(), cv2.RETR_TREE ,  cv2.CHAIN_APPROX_SIMPLE)
Contours = imutils.grab_contours(KeyPoints)
Contours = sorted(Contours , key = cv2.contourArea ,reverse = True)[:10]
location = None 
for contor in Contours :
    approx  = cv2.approxPolyDP(contor, 10, True)
    if len(approx) == 4:
        location = approx
        break
print(location)    
mask = np.zeros(gray.shape , np.uint8)
new_image = cv2.drawContours(mask, [location],0 , 255,-1)
new_image = cv2.bitwise_and(img, img , mask = mask)
plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR ))
(x,y) = np.where(mask==255)
(x1,y1) = (np.min(x) , np.min(y))
(x2,y2) = (np.max(x) , np.max(y))
cropped_image = gray[x1:x2+1 , y1:y2+1 ]
plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR ))
 
reader = easyocr.Reader(["en"])
result = reader.readtext(cropped_image)
print('-------------------------------------------------')
print(result)
# datetime object containing current date and time
now = datetime.now()
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
dt_string =  dt_string +'.csv'
# open the file in the write mode
f = open('q', 'w+')

# create the csv writer
writer = csv.writer(f)

# write a row to the csv file
writer.writerow(result)

# close the file
f.close()