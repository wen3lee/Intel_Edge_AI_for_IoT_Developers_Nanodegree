'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
import cv2

from openvino.inference_engine import IENetwork, IECore

class ModelFaceDetection:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, device='CPU', threshold=0.9):
        '''
        TODO: Use this to set your instance variables.
        '''
        self.model_weights = model_name+'.bin'
        self.model_structure = model_name+'.xml'
        self.device = device
        self.threshold = threshold

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
        cropped_face = self.preprocess_output(image, outputs[self.output_name])

        return cropped_face

    def check_model(self):
        pass

    def preprocess_input(self, image):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        # debug
        #print("image.shape:{}".format(image.shape))
        #print("self.input_shape:{}".format(self.input_shape))

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
        #coords = []

        # output shape: [1, 1, N, 7]
        for box in outputs[0][0]:

            # only keep probability greater than the threshold
            if box[2] >= self.threshold:
                xmin = int(box[3] * self.width)
                ymin = int(box[4] * self.height)
                xmax = int(box[5] * self.width)
                ymax = int(box[6] * self.height)

                #coords.append([xmin, ymin, xmax, ymax])

                cropped_face = image[ymin:ymax, xmin:xmax]

                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)

        #return coords
        return cropped_face

    def initial_size(self, width, height):
        self.width = width
        self.height = height


