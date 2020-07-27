'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
import cv2

from openvino.inference_engine import IENetwork, IECore

class ModelFacialLandmarksDetection:
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

        self.input_name = next(iter(self.model.inputs))
        self.input_shape = self.model.inputs[self.input_name].shape
        self.output_name = next(iter(self.model.outputs))
        self.output_shape = self.model.outputs[self.output_name].shape

    def load_model(self):
        '''
        TODO: You will need to complete this method.
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
        self.net = self.core.load_network(network=self.model, device_name=self.device, num_requests=1)

    def predict(self, image):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        frame = self.preprocess_input(image)
        outputs = self.net.infer({self.input_name:frame})

        # debug
        #print("outputs:{}".format(outputs))

        return self.preprocess_output(image, outputs[self.output_name][0])

    def check_model(self):
        pass

    def preprocess_input(self, image):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        # input shape: BxCxHxW    
        height = self.input_shape[2]
        width = self.input_shape[3]

        image = cv2.resize(image, (width, height))
        image = image.transpose((2,0,1))
        image = image.reshape(1, 3, height, width)

        return image

    def preprocess_output(self, image, outputs):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        # output shape: [1, 10], containing a row-vector of 10 floating point values for five landmarks coordinates

        # debug
        #print("image.shape:{}".format(image.shape))
        #print("x0:{}, y0:{}".format(outputs[0][0][0], outputs[1][0][0]))
        #print("x1:{}, y1:{}".format(outputs[2][0][0], outputs[3][0][0]))

        height, width, channel = image.shape

        # left eye
        left_eye_xmin = int(outputs[0][0][0] * width - 20)
        left_eye_xmax = int(outputs[0][0][0] * width + 20)
        left_eye_ymin = int(outputs[1][0][0] * height - 20)
        left_eye_ymax = int(outputs[1][0][0] * height + 20)

        left_eye_image = image[left_eye_ymin:left_eye_ymax, left_eye_xmin:left_eye_xmax]
        left_eye_center = int((left_eye_xmin + left_eye_xmax) / 2), int((left_eye_ymin + left_eye_ymax) / 2)
        cv2.rectangle(image, (left_eye_xmin, left_eye_ymin), (left_eye_xmax, left_eye_ymax), (0, 0, 255), 1)

        # right eye
        right_eye_xmin = int(outputs[2][0][0] * width - 20)
        right_eye_xmax = int(outputs[2][0][0] * width + 20)
        right_eye_ymin = int(outputs[3][0][0] * height - 20)
        right_eye_ymax = int(outputs[3][0][0] * height + 20)

        right_eye_image = image[right_eye_ymin:right_eye_ymax, right_eye_xmin:right_eye_xmax]
        right_eye_center = int((right_eye_xmin + right_eye_xmax) / 2), int((right_eye_ymin + right_eye_ymax) / 2)
        cv2.rectangle(image, (right_eye_xmin, right_eye_ymin), (right_eye_xmax, right_eye_ymax), (0, 0, 255), 1)

        # debug
        #print("left_eye_center:{}, right_eye_center:{}".format(left_eye_center, right_eye_center))

        return left_eye_image, right_eye_image, left_eye_center, right_eye_center
