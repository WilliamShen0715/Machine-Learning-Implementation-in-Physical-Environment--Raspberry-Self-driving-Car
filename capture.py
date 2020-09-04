# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import cv2
import numpy as np
#import socket
#tcpServerSocket=socket.socket()
#    
#host = '192.168.43.88'
#port=12345
#tcpServerSocket.bind((host,port))
#tcpServerSocket.listen(5)
#c, addr = tcpServerSocket.accept() 
#c_msg = c.recv(4096).decode()
#print(c_msg)
# 選擇第二隻攝影機
#cap = cv2.VideoCapture(1)
vision_data = [] #[牆數,車頭與牆夾角,[車牆距離],[車子資訊],[牆資訊]]
while(True):
  # 從攝影機擷取一張影像
#    ret, img = cap.read()
    img = cv2.imread('wall.jpg')[150:1000,:1800,:]
    #cv2.imshow('original', img)

    rows,cols,ch = img.shape
#start
    #cv2.namedWindow('original')
    img_part = img[:, :, :]
    blurred = cv2.GaussianBlur(img_part, (3, 3), 0)
    img_g = cv2.cvtColor(blurred,cv2.COLOR_BGR2GRAY)
    #cv2.namedWindow('blurred')
    #cv2.imshow('blurred', blurred)
#%%    
#wall
    threshold=898
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    #hsv = cv2.cvtColor(img_g, cv2.COLOR_BGR2HSV)
    #white_mask = cv2.inRange(hsv, (78,43,46), (99,255,255))
    white_mask = cv2.inRange(hsv, (0,0,138), (180,10,255))
    white_masks = np.repeat(white_mask[:, :, np.newaxis], 3, axis=2)
    img_masked1 = img_part.copy()
    #img_masked[white_masks == 0] = 0
    
    contours, hierarchy = cv2.findContours(white_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    rec = []
    area = []   
    rotate=[]
    walls = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        box = cv2.minAreaRect(contour)
        rotate.append(box)
        #print('x={}, y={}, width={}, height={}'.format(x, y, w, h))
        #cv2.rectangle(img_masked, (x,y), (x+w, y+h), (0,255,0), 2)
        rec.append([x,y,w,h])
        area.append(w*h)
    Max=0
    #print("area lengh:{}".format(len(area)))
    if len(area)==0:
        print("not catching box")
        continue
    elif len(area)>1:
        for i in range(1,len(area)):
            if area[i]>threshold: #area[Max]: 
                Max = i
                #cv2.rectangle(img_masked1, (rec[Max][0],rec[Max][1]), (rec[Max][0]+rec[Max][2], rec[Max][1]+rec[Max][3]), (0,255,0), 2)
                selected = np.int64(cv2.boxPoints(rotate[Max]))
                box1_info = rotate[Max]
                cv2.drawContours(img_masked1,[selected],0, (0,255,0), 2)
                print('x={}, y={}, width={}, height={}, area={},angle={}'.format(rec[Max][0],rec[Max][1],rec[Max][2],rec[Max][3],abs(rec[Max][2]*rec[Max][3]),rotate[Max][2]))
                theta_back = rotate[Max][2]
                walls.append(rec[Max])
#%%
#car
    #white_mask = cv2.inRange(hsv, (26,130,160), (34,255,255))
    white_mask = cv2.inRange(hsv, (78,43,46), (99,255,255))
    white_masks = np.repeat(white_mask[:, :, np.newaxis], 3, axis=2)
    img_masked2 = img_part.copy()
    #img_masked[white_masks == 0] = 0
    
    contours, hierarchy = cv2.findContours(white_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    rec = []
    area = []
    rotate=[]
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        box = cv2.minAreaRect(contour)
        rotate.append(box)
        #print('x={}, y={}, width={}, height={}'.format(x, y, w, h))
        #cv2.rectangle(img_masked, (x,y), (x+w, y+h), (0,255,0), 2)
        rec.append([x,y,w,h])
        area.append(w*h)
    Max=0
    for i in range(1,len(area)):
        if area[i]>area[Max]: Max = i
    #cv2.rectangle(img_masked, (x,y), (x+w, y+h), (0,255,0), 2)
    #cv2.rectangle(img_masked2, (rec[Max][0],rec[Max][1]), (rec[Max][0]+rec[Max][2], rec[Max][1]+rec[Max][3]), (0,255,0), 2)
    selected = np.int64(cv2.boxPoints(rotate[Max]))
    car_info = rotate[Max]
    cv2.drawContours(img_masked1,[selected],0, (0,255,0), 2)
    print('x={}, y={}, width={}, height={}, theta={}'.format(rotate[Max][0][0],rotate[Max][0][1],rotate[Max][1][0],rotate[Max][1][1],rotate[Max][2]))
#%%
#car head
    white_mask1 = cv2.inRange(hsv, (0,43,46), (15,255,255))
    white_mask2 = cv2.inRange(hsv, (179,43,46), (180,255,255))
    white_mask = (white_mask1)|(white_mask2)
    white_masks = np.repeat(white_mask[:, :, np.newaxis], 3, axis=2)
    img_masked2 = img_part.copy()
    #img_masked[white_masks == 0] = 0
    
    contours, hierarchy = cv2.findContours(white_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    rec = []
    area = []
    rotate=[]
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        box = cv2.minAreaRect(contour)
        rotate.append(box)
        #print('x={}, y={}, width={}, height={}'.format(x, y, w, h))
        #cv2.rectangle(img_masked, (x,y), (x+w, y+h), (0,255,0), 2)
        rec.append([x,y,w,h])
        area.append(w*h)
    Max=0
    for i in range(1,len(area)):
        if area[i]>area[Max]: Max = i
    #cv2.rectangle(img_masked, (x,y), (x+w, y+h), (0,255,0), 2)
    #cv2.rectangle(img_masked2, (rec[Max][0],rec[Max][1]), (rec[Max][0]+rec[Max][2], rec[Max][1]+rec[Max][3]), (0,255,0), 2)
    selected = np.int64(cv2.boxPoints(rotate[Max]))
    head_info = rotate[Max]
    cv2.drawContours(img_masked1,[selected],0, (0,255,0), 2)
    print('x={}, y={}, width={}, height={}, theta={}'.format(rotate[Max][0][0],rotate[Max][0][1],rotate[Max][1][0],rotate[Max][1][1],rotate[Max][2]))
#%%
    import math
    Max = 0
    while len(walls)>2:
        for i in range(1,len(walls)):
            if (car_info[0][0]-(walls[i][0]+walls[i][2]/2))**2+(car_info[0][1]-(walls[i][1]+walls[i][3]/2))**2>(car_info[0][0]-(walls[Max][0]+walls[Max][2]/2))**2+(car_info[0][1]-(walls[Max][1]+walls[Max][3]/2))**2: Max = i
        del walls[Max]
        Max=0
    angle = head_info[2]-theta_back
    if len(walls)==1:
        dis = math.sqrt((car_info[0][0]-(walls[0][0]+walls[0][2]/2))**2+(car_info[0][1]-(walls[0][1]+walls[0][3]/2))**2)
        vision_data.append(1)
        vision_data.append(angle)
        vision_data.append(dis)
    elif len(walls)==2:
        dis1 = math.sqrt((car_info[0][0]-(walls[0][0]+walls[0][2]/2))**2+(car_info[0][1]-(walls[0][1]+walls[0][3]/2))**2)
        dis2 = math.sqrt((car_info[0][0]-(walls[1][0]+walls[1][2]/2))**2+(car_info[0][1]-(walls[1][1]+walls[1][3]/2))**2)
        vision_data.append(2)
        vision_data.append(angle)
        vision_data.append([dis1,dis2])
    else:
        vision_data.append(0)
        vision_data.append(angle)
    vision_data.append(car_info)
    vision_data.append(head_info)
    vision_data.append(walls)
    cv2.namedWindow('masked')
    cv2.imshow('masked', img_masked1|img_masked2)
    #cv2.imshow('masked', img_g)
#%%
#    while True:
#    	say = input("輸入你想傳送的訊息：")
#    	c.send(say.encode())

#%%
# 若按下 q 鍵則離開迴圈
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
# 釋放攝影機
#cap.release()

# 關閉所有 OpenCV 視窗
cv2.destroyAllWindows()