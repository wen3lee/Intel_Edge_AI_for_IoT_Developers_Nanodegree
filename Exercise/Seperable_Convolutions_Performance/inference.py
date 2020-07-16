from openvino.inference_engine import IENetwork, IECore

import numpy as np
import time

# Loading model
model_path='sep_cnn/sep_cnn'
model_weights=model_path+'.bin'
model_structure=model_path+'.xml'

# TODO: Load the model
core = IECore()
model = IENetwork(model_structure, model_weights)
net = core.load_network(network=model, device_name='CPU')

input_name=next(iter(net.inputs))

# Reading and Preprocessing Image
input_img=np.load('image.npy')
input_img=input_img.reshape(1, 28, 28)

input_dict={input_name:input_img}

# TODO: Using the input image, run inference on the model for 10 iterations
start=time.time()

for _ in range(10):
    net.infer(input_dict)

# TODO: Finish the print statement
print("Time taken to run 10 iterations is: {}".format(time.time()-start))