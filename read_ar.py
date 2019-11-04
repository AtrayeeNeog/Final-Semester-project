#!/usr/bin/python
import serial
import syslog
import time

#The following line is for serial over GPIO
port = '/dev/ttyACM0'


ard = serial.Serial(port,9600,timeout=5)

f= open('ar_data.txt',"a+")

while (True):
    # Serial read section
    msg = ard.readline()
    #print ("Message from arduino: ")
    print (msg)
    f.write(msg + '\n')


