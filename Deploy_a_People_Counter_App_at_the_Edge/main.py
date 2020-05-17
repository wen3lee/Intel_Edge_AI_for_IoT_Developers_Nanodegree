"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import os
import sys
import time
import socket
import json
import cv2

import logging as log
import paho.mqtt.client as mqtt
import numpy as np

from argparse import ArgumentParser
from inference import Network

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60


def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser

def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)

    return client

def preprocessing(input_image, height, width):
    image = cv2.resize(input_image, (width, height))
    image = image.transpose((2,0,1))
    image = image.reshape(1, 3, height, width)
    
    return image

def draw_boxes(frame, result, width, height, prob_threshold):    
    current_count = 0
    people_leaving = 0
    
    for box in result[0][0]:
        confidence = box[2]
        if confidence >= prob_threshold:
            # 1: person
            if 1 == box[1]:
                current_count += 1
                xmin = int(box[3] * width)
                ymin = int(box[4] * height)
                xmax = int(box[5] * width)
                ymax = int(box[6] * height)
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)
                
                if box[5] >= 0.9:
                    people_leaving = 1
            
    return current_count, people_leaving, frame

def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    single_image_mode = False
    last_count = 0
    total_count = 0
    start_time  = 0
    duration = 0
   
    last_status = 0
    continue_status = 0
    # 0: init state 1: people in 2: people leaving 3: people out 
    status = 0
    
    # calculate inference time
    infer_start_time = 0
    
    # calculate accuracy
    people_detect = 0
    People_present = 0
    
    # Initialise the class
    infer_network = Network()
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold

    ### TODO: Load the model through `infer_network` ###
    infer_network.load_model(args.model, args.device, args.cpu_extension)
    net_input_shape = infer_network.get_input_shape()
    
    ### TODO: Handle the input stream ###
    # camera
    if args.input == 'CAM':
        input_stream = 0
    # picture
    elif args.input.endswith('.jpg') or args.input.endswith('.bmp'):
        single_image_mode = True
        input_stream = args.input
    # video
    else:
        input_stream = args.input
   
    cap = cv2.VideoCapture(input_stream)
    
    if input_stream:
        cap.open(args.input)
        
    if not cap.isOpened():
        print("Unable to open input stream.")

    width = int(cap.get(3))
    height = int(cap.get(4))

    ### TODO: Loop until stream is over ###
    while cap.isOpened():

        ### TODO: Read from the video capture ###
        flag, frame = cap.read()
        if not flag:
            break
        
        ### TODO: Pre-process the image as needed ###
        image = preprocessing(frame, net_input_shape[2], net_input_shape[3])
        
        ### TODO: Start asynchronous inference for specified request ###
        # calculate inference time
        infer_start_time = time.time()
        
        infer_network.exec_net(image)
  
        ### TODO: Wait for the result ###
        if infer_network.wait() == 0:
            
            ### TODO: Get the results of the inference request ###
            result = infer_network.get_output()
        
            # calculate inference time
            # client.publish("person", json.dumps({"count": (time.time() - infer_start_time)}))
            
            ### TODO: Extract any desired stats from the results ###
            current_count, people_leaving, frame = draw_boxes(frame, result, width, height, prob_threshold) 
            
            # Calculate consecutive frames to filter detection errors
            if status == 0:
                if current_count > last_count:
                    # people in
                    status = 1
                    last_count = current_count
                    
                    start_time = time.time()

                    #client.publish("person", json.dumps({"count": current_count}))
                    
                    # increase count
                    total_count = current_count - last_count
                    client.publish("person", json.dumps({"total": total_count}))
            elif status == 1:
                
                # calculate accuracy
                people_detect += current_count
                People_present += 1
                
                if people_leaving == 1:
                    
                    # people is leaving
                    status = 2
            elif status == 2:
                if current_count < last_count:
                    if continue_status < 15:
                        continue_status += 1
                    else:
                        continue_status = 0

                        # people out
                        status = 0

                        last_count = current_count

                        duration = time.time() - start_time 
                        
                        client.publish("person/duration", json.dumps({"duration": duration}))
                else:
                    continue_status = 0
            else:
                status = 0
            
            #print("status: {}".format(status))
            
            client.publish("person", json.dumps({"count": current_count}))
            
            # calculate accuracy
            #if People_present != 0:
            #    client.publish("person", json.dumps({"count": (people_detect/People_present*100)}))
            
            ### TODO: Calculate and send relevant information on ###
            ### current_count, total_count and duration to the MQTT server ###
            ### Topic "person": keys of "count" and "total" ###
            ### Topic "person/duration": key of "duration" ###
            
        ### TODO: Send the frame to the FFMPEG server ###
        sys.stdout.buffer.write(frame)
        sys.stdout.flush()

        ### TODO: Write an output image if `single_image_mode` ###
        if single_image_mode: 
            cv2.imwrite('output.jpg', frame)    

    # Release the out writer, capture, and destroy any OpenCV windows
    out.release()
    cap.release()
    cv2.destroyAllWindows()

def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)

if __name__ == '__main__':
    main()
