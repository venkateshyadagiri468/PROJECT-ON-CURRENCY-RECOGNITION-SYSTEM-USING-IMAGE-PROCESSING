http://api.exchangeratesapi.io/v1/latest?access_key=61a1804a6c933edd9a9f280f7f54ab69&base=INR
https://manage.exchangeratesapi.io/dashboard
import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

for i in range(1,7):
    img_rgb = cv2.imread('testImages/'+str(i)+'.jpg')
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    maxValue = 0
    flag = 0
    temps = None
    for root, dirs, directory in os.walk('Dataset'):
        for j in range(len(directory)):
            name = os.path.basename(root)
            if 'Thumbs.db' not in directory[j]:
                template = cv2.imread(root+"/"+directory[j],0)
                w, h = template.shape[::-1]
                img_gray = cv2.resize(img_gray,(w,h))
                img_rgb = cv2.resize(img_rgb,(w,h))
                res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
                threshold = 0.4
                loc = np.where( res >= threshold)
                for pt in zip(*loc[::-1]):
                    #cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
                    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(res)
                    if minVal > maxValue:
                        maxValue = minVal
                        print(root+"/"+directory[j]+" "+str(minVal)+" "+str(maxVal)+" "+str(minLoc)+" "+str(maxLoc)+" "+str(res.ravel()))
                        (startX, startY) = maxLoc
                        endX = startX + template.shape[1]
                        endY = startY + template.shape[0]
                        cv2.rectangle(img_rgb, (startX, startY), (endX, endY), (255, 0, 0), 3)
                        temps = template
                        flag = 1
    if flag == 1:
        cv2.imshow("original",img_rgb)
        cv2.imshow("matched",temps)
        cv2.waitKey(0)
    else:
        print("not matched")
