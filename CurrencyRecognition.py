from tkinter import *
import tkinter
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
import numpy as np
import os
import cv2
from tkinter import messagebox

main = tkinter.Tk()
main.title("Currency Recognition System Using Image Processing")
main.geometry("1200x1200")


global filename

def uploadImage():
    global filename
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir="testImages")
    text.insert(END,str(filename)+" Dataset Loaded\n\n")
    pathlabel.config(text=str(filename)+" Dataset Loaded\n\n")


def recognizedCurrency():
    text.delete('1.0', END)
    global filename
    img_rgb = cv2.imread(filename)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    maxValue = 0
    flag = 0
    temps = None
    recognized = None
    currency = None
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
                        recognized = os.path.basename(root)
                        currency = os.path.basename(os.path.dirname(root))
    if flag == 1:
        img_rgb = cv2.resize(img_rgb,(600,400))
        cv2.putText(img_rgb, 'Currency Recognized as : '+currency+" "+recognized, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 255), 2)
        text.insert(END, 'Currency Value : '+recognized+"\n")
        text.insert(END, 'Currency Name : '+currency+"\n")
        if currency == 'INR':
            text.insert(END, 'Currency in Euro : '+str(float(recognized) / 83.0)+"\n")
            text.insert(END, 'Currency in USD  : '+str(float(recognized) / 75.0)+"\n")
            cv2.putText(img_rgb, 'Currency in Euro : '+str(float(recognized) / 83.0), (10, 55),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 255), 2)
            cv2.putText(img_rgb, 'Currency in USD  : '+str(float(recognized) / 75.0), (10, 85),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 255), 2)
        if currency == 'USD':
            text.insert(END, 'Currency in Euro : '+str(float(recognized) / 1.11)+"\n")
            text.insert(END, 'Currency in INR  : '+str(float(recognized) * 75.0)+"\n")
            cv2.putText(img_rgb, 'Currency in Euro : '+str(float(recognized) / 1.11), (10, 55),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 255), 2)
            cv2.putText(img_rgb, 'Currency in INR  : '+str(float(recognized) * 75.0), (10, 85),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 255), 2)    
        text.update_idletasks()    
        cv2.imshow("original Image",img_rgb)
        cv2.imshow("matched Image",temps)
        cv2.waitKey(0)
    else:
        messagebox.showinfo("Unable to recognized", "Unable to recognized")


def close():
    main.destroy()

font = ('times', 14, 'bold')
title = Label(main, text='Currency Recognition System Using Image Processing')
title.config(bg='DarkGoldenrod1', fg='black')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=5,y=5)

font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload Currency Image", command=uploadImage)
uploadButton.place(x=50,y=100)
uploadButton.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='brown', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=560,y=100)

matchButton = Button(main, text="Run Template Matching Currency Recognition", command=recognizedCurrency)
matchButton.place(x=50,y=150)
matchButton.config(font=font1)

exitButton = Button(main, text="Exit", command=close)
exitButton.place(x=50,y=200)
exitButton.config(font=font1)

font1 = ('times', 12, 'bold')
text=Text(main,height=25,width=100)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=520,y=150)
text.config(font=font1)


main.config(bg='LightSteelBlue1')
main.mainloop()
