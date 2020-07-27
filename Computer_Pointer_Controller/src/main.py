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
    start_model_load_time=time.time()

    # load model
    class_face_detection = ModelFaceDetection(args.model_face_detection, args.device, args.threshold)
    class_face_detection.load_model()

    class_head_pose_estimation = ModelHeadPoseEstimation(args.model_head_pose_estimation, args.device)
    class_head_pose_estimation.load_model()

    class_facial_landmarks_detection = ModelFacialLandmarksDetection(args.model_facial_landmarks_detection, args.device)
    class_facial_landmarks_detection.load_model()

    class_gaze_estimation = ModelGazeEstimation(args.model_gaze_estimation, args.device)
    class_gaze_estimation.load_model()

    total_model_load_time = time.time() - start_model_load_time

    # input image
    feed=InputFeeder(input_type='video', input_file=args.input)
    feed.load_data()

    # output
    initial_w, initial_h, initial_fps = feed.get_info()

    counter = 0
    start_inference_time = time.time()

    # debug
    #print("initial_w:{}, initial_h:{}, initial_fps:{}".format(initial_w, initial_h, initial_fps))

    #out_video = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), initial_fps, (initial_w, initial_h), True)
    out_video = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 10, (initial_w, initial_h), True)

    class_face_detection.initial_size(initial_w, initial_h)

    #mc = MouseController(precision='low', speed='slow')
    mc = MouseController(precision='high', speed='fast')

    for flag, batch in feed.next_batch():
        if not flag:
            break

        counter += 1

        # debug
        #print("batch.shape:{}".format(batch.shape))
        # if batch is not None:

        # face_detection
        cropped_face = class_face_detection.predict(batch)

        # head_pose_estimation
        head_pose_angles = class_head_pose_estimation.predict(cropped_face)

        # debug
        #print("angle_y_fc:{}, angle_p_fc:{}, angle_r_fc:{}".format(head_pose_angles[0], head_pose_angles[1], head_pose_angles[2]))

        # facial_landmarks_detection
        left_eye_image, right_eye_image, left_eye_center, right_eye_center= class_facial_landmarks_detection.predict(cropped_face)

        # gaze_estimation
        x, y, gaze_vector = class_gaze_estimation.predict(left_eye_image, right_eye_image, head_pose_angles)

        cv2.line(cropped_face, left_eye_center, (int(left_eye_center[0] + gaze_vector[0] * 100), int(left_eye_center[1] - gaze_vector[1] * 100)), (255,255,255), 2)
        cv2.line(cropped_face, right_eye_center, (int(right_eye_center[0] + gaze_vector[0] * 100), int(right_eye_center[1] - gaze_vector[1] * 100)), (255,255,255), 2)

        # output
        cv2.imshow('output', batch)
        cv2.waitKey(30)
        cv2.imwrite('output.jpg', batch);

        out_video.write(batch)

        # MouseController
        mc.move(x, y)

    total_time = time.time() - start_inference_time
    total_inference_time = round(total_time, 1)
    fps = counter/total_inference_time

    print("total_model_load_time:{}, total_inference_time:{}, fps:{}".format(total_model_load_time, total_inference_time, fps))

    feed.close()
    cv2.destroyAllWindows()

if __name__=='__main__':
    parser=ArgumentParser()
    parser.add_argument("-mfd", "--model_face_detection", required=True)
    parser.add_argument("-mhpe", "--model_head_pose_estimation", required=True)
    parser.add_argument("-mfld", "--model_facial_landmarks_detection", required=True)
    parser.add_argument("-mge", "--model_gaze_estimation", required=True)

    parser.add_argument("-d", "--device", default="CPU")
    parser.add_argument("-t", "--threshold", default=0.9)
    parser.add_argument("-i", "--input", required=True)

    args=parser.parse_args() 
    main(args)
