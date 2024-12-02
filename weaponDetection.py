#!/usr/bin/env python3
#
# This inference reuses sample code published in Nvidia dusty-nv github repo 

import sys
#import argparse
import time
import Jetson.GPIO as GPIO

from jetson_inference import detectNet
from jetson_utils import videoSource, videoOutput, Log


# create video sources and outputs
input = videoSource("csi://0")

#output = videoOutput(args.output, argv=sys.argv)
output = videoOutput()
    
net = detectNet(model="models/weapon/ssd-mobilenet.onnx", labels="models/weapon/labels.txt", input_blob="input_0", output_cvg="scores", 
                    output_bbox="boxes", threshold=0.5)

# process frames until EOS or the user exits
# Led pin simulate warning for  weapon detection.
led_pin11 = 11

# Set up the GPIO channel
GPIO.setmode(GPIO.BOARD) 
GPIO.setup(led_pin11, GPIO.OUT, initial=GPIO.LOW)

while True:
    

    # capture the next image
    img = input.Capture()

    if img is None: # timeout
        GPIO.output(led_pin11, GPIO.LOW)
        continue  
        
    # detect objects in the image (with overlay)
    detections = net.Detect(img, overlay="box,labels,conf")

    # print the detections
    print("detected {:d} objects in image".format(len(detections)))

    #for detection in detections:
    #    print(detection)

    #detectionCycle
    unsafePersonel = False
    for detection in detections:
        print(detection)
        classLabel = net.GetClassLabel(detection.ClassID)
        print(f'label: {classLabel}')
        if 'unsafe'.__eq__(classLabel):
          unsafePersonel = True

    # render the image
    output.Render(img)

    # update the title bar
    output.SetStatus("{:s} | Network {:.0f} FPS".format("ssd-mobilenet-v2", net.GetNetworkFPS()))
    
    if unsafePersonel == True:
        print("Insufficient safety gear detected.")
        GPIO.output(led_pin11, GPIO.HIGH) 
        print("LED11 is ON")
    else:
        GPIO.output(led_pin11, GPIO.LOW)
        print("LED11 is OFF")

    #time.sleep(5000)
    #output.Close()

    # print out performance info
    #net.PrintProfilerTimes()

    # exit on input/output EOS
    if not input.IsStreaming() or not output.IsStreaming():
        break