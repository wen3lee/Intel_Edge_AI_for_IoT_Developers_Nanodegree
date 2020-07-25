'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''

from openvino.inference_engine import IENetwork, IECore

class ModelHeadPoseEstimation:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None):
        '''
        TODO: Use this to set your instance variables.
        '''
        self.model_weights=model_name+'.bin'
        self.model_structure=model_name+'.xml'
        self.device=device

        self.core = IECore()
        self.model = self.core.read_network(model=self.model_structure, weights=self.model_weights)

        self.input_name=next(iter(self.model.inputs))
        self.input_shape=self.model.inputs[self.input_name].shape
        self.output_name=next(iter(self.model.outputs))
        self.output_shape=self.model.outputs[self.output_name].shape

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
        coords = self.preprocess_outputs(outputs[self.output_name])

    def check_model(self):
        pass

    def preprocess_input(self, image):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        # input shape: 1xCxHxW    
        height = self.input_shape[2]
        width = self.input_shape[3]

        image = cv2.resize(image, (width, height))
        image = image.transpose((2,0,1))
        image = image.reshape(1, 3, height, width)

        return image

    def preprocess_output(self, outputs):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        # angle_y_fc: [1, 1] - Estimated yaw
        # angle_p_fc: [1, 1] - Estimated pitch
        # angle_r_fc: [1, 1] - Estimated roll
