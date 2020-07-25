import time
import numpy as np
import cv2
#import os

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

    # load model
    class_face_detection = ModelFaceDetection(args.model_face_detection, args.device, args.threshold)
    class_face_detection.load_model()

    ''' temp mark
    class_head_pose_estimation = ModelHeadPoseEstimation(args.model_head_pose_estimation, args.device)
    class_head_pose_estimation.load_model()

    class_facial_landmarks_detection = ModelFacialLandmarksDetection(args.model_facial_landmarks_detection, args.device)
    class_facial_landmarks_detection.load_model()

    class_gaze_estimation = ModelGazeEstimation(args.model_gaze_estimation, args.device)
    class_gaze_estimation.load_model()
    '''

    # input image
    feed=InputFeeder(input_type='video', input_file=args.input)
    feed.load_data()

    # output
    #VideoWriter writer("output.avi", CV_FOURCC('M', 'J', 'P', 'G'), 25.0, Size(640, 480));
    #cv2.VideoWriter(output_filename, fourcc, fps, self._window_shape)
    #cv2.VideoWriter('output.avi', fourcc, 25, Size(640, 480))

    initial_w, initial_h, initial_fps = feed.get_info()

    # debug
    #print("initial_w:{}, initial_h:{}, initial_fps:{}".format(initial_w, initial_h, initial_fps))

    #out_video = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), initial_fps, (initial_w, initial_h), True)
    out_video = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 10, (initial_w, initial_h), True)
    
    class_face_detection.initial_size(initial_w, initial_h)

    for flag, batch in feed.next_batch():
        if not flag:
            break

        # debug
        #print("batch.shape:{}".format(batch.shape))
        # if batch is not None:

        # face_detection
        coords, image = class_face_detection.predict(batch)

        # output
        #cv2.imshow('output', batch)
        #cv2.waitKey(30)
        #cv2.imwrite('output.jpg', batch);

        out_video.write(batch)

    # MouseController
    mc = MouseController(precision='low', speed='slow')
    #mc.move(10, 10)

    # finish time
    #print("time: {}".format(time.time() - start))

    feed.close()
    cv2.destroyAllWindows()

if __name__=='__main__':
    parser=ArgumentParser()
    parser.add_argument("-mfd", "--model_face_detection", required=True)

    ''' temp mark
    parser.add_argument("-mhpe", "--model_head_pose_estimation", required=True)
    parser.add_argument("-mfld", "--model_facial_landmarks_detection", required=True)
    parser.add_argument("-mge", "--model_gaze_estimation", required=True)
    '''

    parser.add_argument("-d", "--device", default="CPU")
    parser.add_argument("-t", "--threshold", default=0.9)
    parser.add_argument("-i", "--input", required=True)

    args=parser.parse_args() 
    main(args)
