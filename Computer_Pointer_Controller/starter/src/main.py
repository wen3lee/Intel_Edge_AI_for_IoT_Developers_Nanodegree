import time
import numpy as np
import cv2

from argparse import ArgumentParser

from input_feeder import InputFeeder
from mouse_controller import MouseController

from face_detection import ModelFaceDetection
from head_pose_estimation import ModelHeadPoseEstimation
from facial_landmarks_detection import ModelFacialLandmarksDetection
from gaze_estimation import ModelGazeEstimation

def main(args):
    # start time
    start=time.time()
    
    # face_detection
    fd = ModelFaceDetection(args.model, args.device)
    fd.load_model()
    
    # input image
    feed=InputFeeder(input_type='video', input_file=args.input)
    feed.load_data()
    
    for flag, batch in feed.next_batch():
        if not flag:
            break
            
        if batch is not None:
            # debug
            #print("batch.shape:{}".format(batch.shape))
            fd.preprocess_input(batch)
        
    # MouseController
    mc = MouseController(precision='low', speed='slow')
    #mc.move(10, 10)
    
    # finish time    
    #print("time: {}".format(time.time() - start))
    
    feed.close()
        
    
if __name__=='__main__':
    parser=ArgumentParser()
    parser.add_argument("-m", "--model", required=True)
    parser.add_argument("-d", "--device", default="CPU")
    parser.add_argument("-i", "--input", required=True)
    
    args=parser.parse_args() 
    main(args)
