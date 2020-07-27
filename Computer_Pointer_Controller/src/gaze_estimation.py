'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
import cv2
import math

from openvino.inference_engine import IENetwork, IECore

class ModelGazeEstimation:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, device='CPU'):
        '''
        TODO: Use this to set your instance variables.
        '''
        self.model_weights=model_name+'.bin'
        self.model_structure=model_name+'.xml'
        self.device=device

        self.core = IECore()
        self.model = self.core.read_network(model=self.model_structure, weights=self.model_weights)

        self.input_name = [i for i in self.model.inputs.keys()]

        # debug
        #print("self.input_name:{}".format(self.input_name))
        #print("self.model.inputs[self.input_name[1]].shape:{}".format(self.model.inputs[self.input_name[1]].shape))

        self.input_shape=self.model.inputs[self.input_name[1]].shape
        self.output_name=next(iter(self.model.outputs))
        self.output_shape=self.model.outputs[self.output_name].shape

    def load_model(self):
        '''
        TODO: You will need to complete this method.
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
        self.net = self.core.load_network(network=self.model, device_name=self.device, num_requests=1)

    def predict(self, left_eye_image, right_eye_image, head_pose_angles):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        left_eye_frame = self.preprocess_input(left_eye_image)
        right_eye_frame = self.preprocess_input(right_eye_image)

        net_input = {'left_eye_image': left_eye_frame, 'right_eye_image': right_eye_frame, 'head_pose_angles': head_pose_angles}
        outputs = self.net.infer(inputs=net_input)

        # debug
        #print("outputs:{}".format(outputs))

        return self.preprocess_output(outputs, head_pose_angles)

    def check_model(self):
        pass

    def preprocess_input(self, image):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        # head_pose_angles: BxC
        # left_eye_image: BxCxHxW
        # right_eye_image: BxCxHxW

        height = self.input_shape[2]
        width = self.input_shape[3]

        image = cv2.resize(image, (width, height))
        image = image.transpose((2,0,1))
        image = image.reshape(1, 3, height, width)

        return image

    def preprocess_output(self, outputs, head_pose_angles):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        gaze_vector = outputs[self.output_name][0]

        angle_r_fc = head_pose_angles[2]
        r_cos = math.cos(angle_r_fc * math.pi / 180.0)
        r_sin = math.sin(angle_r_fc * math.pi / 180.0)

        x = gaze_vector[0] * r_cos + gaze_vector[1] * r_sin
        y = -gaze_vector[0] * r_sin+ gaze_vector[1] * r_cos

        #debug
        #print("gaze_vector:{}".format(gaze_vector))
        #print("x:{}, y:{}".format(x, y))

        return x, y, gaze_vector
