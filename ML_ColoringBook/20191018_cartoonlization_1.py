# -*- coding: utf-8 -*-

import cv2
import os


def cartoonise(picture_name):

    #capturing image effectively
    imgInput_FileName = picture_name
    edge_filename = 'edge_' + picture_name
    saved_filename = 'cartoon_' + picture_name
    
    num_bilateral = 7   
    print("Cartoonnizing" + imgInput_FileName)
    #read image
    img_rgb = cv2.imread(imgInput_FileName)     
    img_color = img_rgb


    for _ in range(num_bilateral):
        img_color = cv2.bilateralFilter(img_color,d=9,sigmaColor=9,sigmaSpace=7)
       
    #gray and blur
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    img_blur = cv2.medianBlur(img_gray, 7)
    
    #edge detection
    img_edge = cv2.adaptiveThreshold(img_blur,255,
                                     cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY,
                                     blockSize=9,
                                     C=2)
    
    #transfer to color image
    img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2RGB)
    img_cartoon = cv2.bitwise_and(img_color, img_edge)
    
    
    cv2.imwrite(edge_filename , img_edge)
    cv2.imwrite(saved_filename , img_cartoon)

cartoonise('image_001.jpg')
cartoonise('image_002.jpg')
cartoonise('testing001.jpg')
