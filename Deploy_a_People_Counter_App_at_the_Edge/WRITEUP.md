## Explaining Custom Layers

A) download  
python3 downloader.py --name ssd_mobilenet_v2_coco -o /home/workspace/model 

B) converting to Intermediate Representation  
python3 mo_tf.py --input_model  /home/workspace/model/public/ssd_mobilenet_v2_coco/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb --tensorflow_use_custom_operations_config  extensions/front/tf/ssd_v2_support.json --tensorflow_object_detection_api_pipeline_config  /home/workspace/model/public/ssd_mobilenet_v2_coco/ssd_mobilenet_v2_coco_2018_03_29/pipeline.config -o /home/workspace/ir19r3/ssd_mobilenet_v2_coco 

C)  --input_model  /home/workspace/model/public/ssd_mobilenet_v2_coco/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb 
File with a pre-trained model (binary or text .pb file after freezing) 

D) --tensorflow_use_custom_operations_config  extensions/front/tf/ssd_v2_support.json  
A subgraph replacement configuration file that describes rules to convert specific TensorFlow* topologies. 

E) --tensorflow_object_detection_api_pipeline_config  /home/workspace/model/public/ssd_mobilenet_v2_coco/ssd_mobilenet_v2_coco_2018_03_29/pipeline.config 
A special configuration file that describes the topology hyper-parameters and structure of the TensorFlow Object Detection API model. 

F) -o /home/workspace/ir19r3/ssd_mobilenet_v2_coco 
Output path 

## Comparing Model Performance

Test           | Pretrained(Tf) | IR Conversion | Difference 
------------------------------------------------------------
Inference Time | 31 ms          | 74 ms         | 43 ms 

Size of Model  | 63 MB          | 65 MB         | 2 MB 

Accuracy       | 22 %           | 63 %          | 41% 

## Assess Model Use Cases

A) Some of the potential use cases of the people counter app are... 
When I was working on this project, I told my son that I would write a program to monitor his behavior at home. I also want to calculate his time playing video games.  

B) Each of these use cases would be useful because... 
It seems that everyday life will happen and be used. 

## Assess Effects on End User Needs

A) Lighting: 
Directly affect the accuracy.  

B) Model accuracy: 
In this case, lower accuracy may lead to errors in the total number of people and the length of stay of the people. It may not provide effective advice to the application. 

C) Camera focal length: 
Photography from different directions and angles directly affects accuracy. 

D) Image size:   
This is a trade-off. Larger and precise image size has higher accuracy in theory, but it takes more time to inference. 
