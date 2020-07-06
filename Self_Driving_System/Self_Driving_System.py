import serial
import smbus            #import SMBus module of I2C
import threading
import time
from time import sleep          #import

import cv2
import numpy as np
import math
import json
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.svm import SVC
from picamera import PiCamera
from picamera.array import PiRGBArray
from stereovision.calibration import StereoCalibrator
from stereovision.calibration import StereoCalibration
from datetime import datetime
import requests
from datetime import date
from matplotlib import pyplot as plt

Anomaliesnumber=0
df = pd.read_excel('Anomaly.xlsx', sheet_name='Sheet1')
X = np.array(df.drop(['class'],1))
y = np.array(df['class'])

X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.3)
#Guassian Kernel function
svclassifier = SVC(kernel='rbf' , gamma='scale')
svclassifier.fit(X_train, y_train)
accuracy = svclassifier.score(X_test,y_test)
#print("Accuracy = " , accuracy)
#some MPU6050 Registers and their Address
PWR_MGMT_1   = 0x6B
SMPLRT_DIV   = 0x19
CONFIG       = 0x1A
GYRO_CONFIG  = 0x1B
INT_ENABLE   = 0x38
ACCEL_XOUT_H = 0x3B
ACCEL_YOUT_H = 0x3D
ACCEL_ZOUT_H = 0x3F
GYRO_XOUT_H  = 0x43
GYRO_YOUT_H  = 0x45
GYRO_ZOUT_H  = 0x47

def MPU_Init():
    #write to sample rate register
    bus.write_byte_data(Device_Address, SMPLRT_DIV, 7)
    
    #Write to power management register
    bus.write_byte_data(Device_Address, PWR_MGMT_1, 1)
    
    #Write to Configuration register
    bus.write_byte_data(Device_Address, CONFIG, 0)
    
    #Write to Gyro configuration register
    bus.write_byte_data(Device_Address, GYRO_CONFIG, 24)
    
    #Write to interrupt enable register
    bus.write_byte_data(Device_Address, INT_ENABLE, 1)

def read_raw_data(addr):
    #Accelero and Gyro value are 16-bit
        high = bus.read_byte_data(Device_Address, addr)
        low = bus.read_byte_data(Device_Address, addr+1)
    
        #concatenate higher and lower value
        value = ((high << 8) | low)
        
        #to get signed value from mpu6050
        if(value > 32768):
                value = value - 65536
        return value


bus = smbus.SMBus(1)    # or bus = smbus.SMBus(0) for older version boards
Device_Address = 0x68   # MPU6050 device address

MPU_Init()

def ReadAndClassify():
    while(True):
        acc_x = read_raw_data(ACCEL_XOUT_H)
        acc_y = read_raw_data(ACCEL_YOUT_H)
        acc_z = read_raw_data(ACCEL_ZOUT_H)
        global Anomaliesnumber
        
        #Read Gyroscope raw value
        gyro_x = read_raw_data(GYRO_XOUT_H)
        gyro_y = read_raw_data(GYRO_YOUT_H)
        gyro_z = read_raw_data(GYRO_ZOUT_H)
        
        #Full scale range +/- 250 degree/C as per sensitivity scale factor
        Ax = acc_x/16384.0
        Ay = acc_y/16384.0
        Az = acc_z/16384.0
        
        Gx = gyro_x/131.0
        Gy = gyro_y/131.0
        Gz = gyro_z/131.0
        
        xslGx=format(Gx,'.2f')
        xslGy=format(Gy,'.2f')
        xslGz=format(Gz,'.2f')
        xslAx=format(Ax,'.2f')
        xslAy=format(Ay,'.2f')
        xslAz=format(Az,'.2f')
        global stop
        
        example = np.array([xslGx, xslGy, xslGz, xslAx, xslAy, xslAz])  # LIST OF LISTS
        example = example.reshape(1,-1)
        prediction = svclassifier.predict(example)
        if (prediction == 0):
            prediction=0 #print("Clear Road!")
        elif (prediction == 1):
            Anomaliesnumber = Anomaliesnumber+1
            print("Anomaly Detected!")
            #break
        sleep(0.5)
      
    
def load_map_settings( fName ):
    f=open(fName, 'r')
    data = json.load(f)
    sbm.setPreFilterType(1)
    sbm.setPreFilterSize(data['preFilterSize'])
    sbm.setPreFilterCap(data['preFilterCap'])
    sbm.setMinDisparity(data['minDisparity'])
    sbm.setNumDisparities(data['numberOfDisparities'])
    sbm.setTextureThreshold(data['textureThreshold'])
    sbm.setUniquenessRatio(data['uniquenessRatio'])
    sbm.setSpeckleRange(data['speckleRange'])
    sbm.setSpeckleWindowSize(data['speckleWindowSize']    )
    f.close()

sbm = cv2.StereoBM_create(numDisparities=0, blockSize=21)
threading.Thread(target=load_map_settings ("3dmap_set.txt")).start()

calibration = StereoCalibration(input_folder='calib_result')
stop = False
img_width = int ((int((1280+31)/32)*32) * 0.5)
img_height = int ((int((480+15)/16)*16) * 0.5)
capture = np.zeros((img_height, img_width, 4), dtype=np.uint8)
camera = PiCamera(stereo_mode='side-by-side',stereo_decimate=False)
camera.resolution=((int((1280+31)/32)*32), (int((480+15)/16)*16))
camera.framerate = 20
camera.vflip = True
#ser = serial.Serial('/dev/ttyACM0', 9600, timeout=0)
#ser.flush()

disparity = np.zeros((img_width, img_height), np.uint8)
theta=0
minLineLength = 5
maxLineGap = 10

def stereo_depth_map(rectified_pair):
    dmLeft = rectified_pair[0]
    dmRight = rectified_pair[1]
    disparity = sbm.compute(dmLeft, dmRight)
    local_max = disparity.max()
    local_min = disparity.min()
    disparity_grayscale = (disparity-local_min)*(65535.0/(local_max-local_min))
    disparity_color = cv2.convertScaleAbs(disparity_grayscale, alpha=(255.0/65535.0))
    cv2.imshow("Disparity", disparity_color)
    cv2.imshow("Image", dmRight)

    
    key = cv2.waitKey(1) & 0xFF   
    if key == ord("q"):
        quit();
    return disparity_color
def destance_calc(imgLeft, imgRight):
    global stop
    rectified_pair = calibration.rectify((imgLeft, imgRight))
    disparity = stereo_depth_map(rectified_pair)
    hist = cv2.calcHist([disparity],[0],None,[250],[1,250])
    for px in range(0,240):
        if (hist[px]>[200]):
            if (px + 1 < 50):
                print("Stop Moving!!")
                #ser.write(b"0")
                stop = True
                break
            elif(px + 1 < 140):
                print("Slow Down!")
                #ser.write(b"1")
                break
            #else:
                #print("Speed Up!")
                #ser.write(b"2")
                #break
             
def roadLaneDetection(image):
   #image = image[200:640,0:635]
   blurred = cv2.GaussianBlur(image, (5, 5), 0)
   edged = cv2.Canny(blurred, 85, 85)
   lines = cv2.HoughLinesP(edged,1,np.pi/180,10,minLineLength,maxLineGap)
   if np.any(lines !=None):
       global theta
       for x in range(0, len(lines)):
           for x1,y1,x2,y2 in lines[x]:
               cv2.line(image,(x1,y1),(x2,y2),(0,0,255),2)
               theta=theta+math.atan2((y2-y1),(x2-x1))
   threshold=8
   if(theta>threshold):
      #ser.write(b"L")
      print("left")
   elif(theta<-threshold):
      #ser.write(b"R")
      print("Right")
   #elif(abs(theta)==0):        
       #ser.write(b"S")
       #print("STOP")
   #else:
       #ser.write(b"F")
       #print("Forward")    
   theta=0
   cv2.imshow("Roadlane",image)

now = datetime.now()
today = date.today()
starttime = now.strftime("%H:%M:%S")
threading.Thread(target=ReadAndClassify).start()
for frame in camera.capture_continuous(capture, format="bgra", use_video_port=True, resize=(img_width,img_height)):
    #t1 = datetime.now()
    pair_img = cv2.cvtColor (frame, cv2.COLOR_BGR2GRAY)
    imgLeft = pair_img [0:img_height,0:int(img_width/2)] #Y+H and X+W
    imgRight = pair_img [0:img_height,int(img_width/2):img_width] #Y+H and X+W
    destance_calc(imgLeft, imgRight)
    
    threading.Thread(target=roadLaneDetection(imgRight)).start()
    


    
    #t2 = datetime.now()
    #print ("DM build time: " + str(t2-t1))
    if(stop==True):
        now2 = datetime.now()
        endtime = now2.strftime("%H:%M:%S")
        reportdata = {'Numofanomalies': '1','Tripstart': starttime,
              'Tripend': endtime,'Date': today}
        resp = requests.post('https://selfdrivingcarserver.000webhostapp.com/pyreportinsert.php',
                     data=reportdata)
        exit()


        
    
    
