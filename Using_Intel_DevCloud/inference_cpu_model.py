
import time
import numpy as np
import cv2
from openvino.inference_engine import IENetwork
from openvino.inference_engine import IECore
import argparse

def main(args):
    model=args.model_path
    model_weights=model+'.bin'
    model_structure=model+'.xml'
    
    start=time.time()
    
    # TODO: Load the model
    plugin = IECore()
    network = IENetwork(model=model_structure, weights=model_weights)
    exec_network = plugin.loadnetwork(network, device_name='CPU')
    
    print(f"Time taken to load model = {time.time()-start} seconds")
    
    # Get the name of the input node
    input_name=next(iter(model.inputs))
    
    # Reading and Preprocessing Image
    input_img=cv2.imread('/data/resources/car.png')
    input_img=cv2.resize(input_img, (300,300), interpolation = cv2.INTER_AREA)
    input_img=np.moveaxis(input_img, -1, 0)

    # TODO: Prepare the model for inference (create input dict etc.)
    print(network.inputs)
    #input_blob = next(iter(network.inputs)
    #                  input_img
    
    start=time.time()
    for _ in range(10):
        # TODO: Run Inference in a Loop
        exec_network.requests[0].infer({input_blob: image})
    
    print(f"Time Taken to run 10 Infernce on CPU is = {time.time()-start} seconds")

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    
    args=parser.parse_args() 
    main(args)