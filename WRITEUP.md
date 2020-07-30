# Project Write-Up

 The people counter application is a Ai IoT solution that can detect people in an areas.
 Providing the number, average duration, total count of people since the start of the observation session and an informations sends to the  UI telling the user when a person enters the video frame.

In this project I used 'person-detection-retail-0013'.


## Explaining Custom Layers
- Custom layers are the layers that are not included to the list of known layers.
- Model optimizer compare each layer of the input model in the list of known layers before building the model's internal representation, optimizing the model, and producing the Intermediate Representation. If model contains any layer that are not in the list of known layers, that is are custom layers.

For supported layers visit this site: https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_Supported_Frameworks_Layers.html


Some of the potential reasons for handling custom layers are...

- When a layer isn’t supported by the Model Optimizer ,Model Optimizer does not know about the custom layers.
- Custom layers needs to handle. Because without handling it model optimizer can not convert specific model to Intermediate Representation.



## Comparing Model Performance

My method(s) to compare models before and after conversion to Intermediate Representations
were:

The difference between model accuracy pre- and post-conversion was...
- Accuracy of the pre-conversion model not bad not good.
- Accuracy of the post-conversion model was good.

The size of the model pre- and post-conversion was...
- The size of the fozen inference graph.pb = 69.5 Mb and size of the pos-conversion model xml+bin file = 64.3 5Mb

The inference time of the model pre- and post-conversion was...
- I mentioned the information in the model section about the inference times.

The CPU Overhead of the model pre- and post-conversion was...
-CPU overhead of the pre conversion model around 65% per core.
-CPU overhead of the post conversion model around 40% per core.

Compare the differences in network needs and costs of using cloud services as opposed to deploying at the edge...
- Edge model needs only local network connection. It does not require high networkk speeds like cloud.
- Cloud services prices is so high. Where edge model can run on minimal cpu with local network connection. It is saves money and time.

## Assess Model Use Cases

Some of the potential use cases of the people counter app are...
1. Security control
2. Visitor analysis for shops
3. Mask is wearing or not
4. Social distancing
5. Detect people trying to inter restricted area

Each of these would be useful.

## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a
deployed edge model. The potential effects of each of these are as follows...

- Ligtning: We need place lights because model can't predict so accurately when object is dark.
- Model accuracy: Better the model accuracy, More are the chances to obtain the desired results through an app deployed at edge.
- Camera/Image focal length: The better the image's pixel quality or camera focal length, the goo the results.
- Image size: The better the image resolution, the better the result. 

## Model Research

In investigating potential people counter models, I tried each of the following three models:

- Model 1: [ ssd_mobilenet_v2_coco_2018_03_29 ]
  - [ http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz ]
  - Location on workspace
      ```
    /home/workspace/ssd_mobilenet_v2_coco_2018_03_29
      ```
    
- I converted the model to an Intermediate Representation with the following arguments... 
    ```
    cd /opt/intel/openvino/deployment_tools/model_optimizer
    ```
    ```
    python mo_tf.py --input_model /home/workspace/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb --tensorflow_use_custom_operations_config extensions/front/tf/ssd_v2_support.json --tensorflow_object_detection_api_pipeline_config /home/workspace/ssd_mobilenet_v2_coco_2018_03_29/pipeline.config --reverse_input_channels -o /home/workspace/ssd_mobilenet_v2_coco_2
    ```
- I used this model using...
    ```  
    cd /opt/workspace/
    ``` 
    ```
    python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm
    ```   

- The model was not bad for the app but still insufficient because...
- We can get better results using other models

- Avarage Duration: 00:02 sc
- Inference Time: 67.586 ms


- I tried to improve the model for the app by...
- Changing the threshold to 0.40. Sometimes miss person when person wear dark clothes and drawing the boxes around a person.

  
  
- Model 2: [ faster_rcnn_inception_v2_coco_2018_01_28 ]
  - [ http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz ]

  - Location on workspace
      ```
    /home/workspace/faster_rcnn_inception_v2_coco_2018_01_28
      ```
 
  - I converted the model to an Intermediate Representation with the following arguments...
    ```
    cd /opt/intel/openvino/deployment_tools/model_optimizer
    ```
    ```
    python mo_tf.py --input_model /home/workspace/faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb --tensorflow_use_custom_operations_config extensions/front/tf/ssd_v2_support.json --tensorflow_object_detection_api_pipeline_config /home/workspace/faster_rcnn_inception_v2_coco_2018_01_28/pipeline.config --reverse_input_channels -o /home/workspace/faster_rcnn_inception_v2_coco_2018_01_28
    ```
  - I used this model using...
    ```  
    cd /opt/workspace/
    ``` 
    ```
    python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm

    ```
  - The model was insufficient for the app because...
  - Almost same results when we use a "ssd_mobilenet_v2_coco" model. We can get better results using other models.

  - Avarage Duration: 00:02 sc
  - Inference Time: 70.236 ms


  - I tried to improve the model for the app by...
  - Changing the threshold to 0.40. App run slow and sometimes miss person and drawing the boxes around a person.


- Model 3: [ person-detection-retail-0013 ]
  - [ https://docs.openvinotoolkit.org/2018_R5/_docs_Retail_object_detection_pedestrian_rmnet_ssd_0013_caffe_desc_person_detection_retail_0013.html ]

  - Location on workspace
      ```
    /home/workspace/person-detection-retail-0013
      ```
 
  - I downloaded the model with the following arguments...
    ```
    cd /opt/intel/openvino/deployment_tools/model_optimizer
    ``` 
    ``` 
    python downloader.py --name person-detection-retail-0013 --precisions FP16 -o /home/workspace/person-detection-retail-0013
    ``` 

  - I used this model using
    ``` 
    python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m person-detection-retail-0013/person-detection-retail-0013.xml  -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm

    ``` 
  - The model was good and sufficient for the app because...
  - Good result on count, duration and inference time on this model.

  - Avarage Duration: 00:10 sc
  - Inference Time: 46.116 ms

  - I tried to improve the model for the app by...
  - Changing the threshold to 0.40. Sometimes miss person but still good.
