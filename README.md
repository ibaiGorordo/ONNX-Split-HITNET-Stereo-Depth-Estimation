# ONNX-Split-HITNET-Stereo-Depth-Estimation
 Python scripts for performing Stereo Depth Estimation using the HITNET model split into two parts in ONNX.

![Hitnet ONNX model split](https://github.com/ibaiGorordo/ONNX-Split-HITNET-Stereo-Depth-Estimation/blob/main/doc/img/split.png)

# Important
- The main objective of this repository is to test the idea of inferencing a model by divining it into two models.
- The reason for trying this idea is for cases in which there are two processors connected (e.g. Intel MyriadX and host computer) so that the load can be shared. Particularly, in cases like the HITNET model where not all operations are supported in the MyriadX, the model can be separated into one part running in the MyriadX and the other in the computer. However,it is very likely that it is faster to run everything in the host device.
- This repository is mainly for "educational" purposes, if you want to use this model in real life, it is better to use the original model (link below).
- Currently, it only works witht the flyingthings version of the HITNET model.

# Installation
```
git clone https://github.com/ibaiGorordo/ONNX-Split-HITNET-Stereo-Depth-Estimation.git
cd ONNX-Split-HITNET-Stereo-Depth-Estimation
pip install -r requirements.txt
```

# ONNX model
- The original model was converted to different formats (including .onnx) by [PINTO0309](https://github.com/PINTO0309), download the models from [his repository](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/142_HITNET) and save them into the **[models](https://github.com/ibaiGorordo/ONNX-Split-HITNET-Stereo-Depth-Estimation/tree/main/models)** folder. 
- To split the model into two, you can use the `create_split_hitnet_model.py` script. Additionally, the code will also run the split automatically if the split has not been done previously.

# Original ONNX Inference
For performing the inference in ONNX with the original model, check my other repository **[ONNX HITNET Stereo Depth estimation](https://github.com/ibaiGorordo/ONNX-HITNET-Stereo-Depth-estimation)**.

# Pytorch inference
For performing the inference in Tensorflow, check my other repository **[HITNET Stereo Depth estimation](https://github.com/ibaiGorordo/HITNET-Stereo-Depth-estimation)**.

# TFLite inference
For performing the inference in TFLite, check my other repository **[TFLite HITNET Stereo Depth estimation](https://github.com/ibaiGorordo/TFLite-HITNET-Stereo-depth-estimation)**.

# Original Tensorflow model
The Tensorflow pretrained model was taken from the [original repository](https://github.com/google-research/google-research/tree/master/hitnet).
 
# Examples

 * **Image inference**:
 
  ```
 python image_estimate_depth.py
 ```
 
 ![Hitnet Split depth estimation](https://github.com/ibaiGorordo/ONNX-Split-HITNET-Stereo-Depth-Estimation/blob/main/doc/img/out.jpg)
 
 *Stereo depth estimation on the cones images from the Middlebury dataset (https://vision.middlebury.edu/stereo/data/scenes2003/)*


# Performance comparison
The comparison in the performance between inferencing with the original model Vs. the split model can be obtained by running:
 
  ```
  python compare_performance.py
  ```
**Avg. Inference time** (Nvidia 1660 Super): 
 - **Original**: 103 ms
 - **Split**: 136 ms

# References:
* Hitnet model: https://github.com/google-research/google-research/tree/master/hitnet
* Original ONNX HITNET Infernece: https://github.com/ibaiGorordo/ONNX-HITNET-Stereo-Depth-estimation
* PINTO0309's model zoo: https://github.com/PINTO0309/PINTO_model_zoo
* PINTO0309's model conversion tool: https://github.com/PINTO0309/openvino2tensorflow
* ONNX GraphSurgeon: https://docs.nvidia.com/deeplearning/tensorrt/onnx-graphsurgeon/docs/index.html
* Original paper: https://arxiv.org/abs/2007.12140
* Depthai Python library: https://github.com/luxonis/depthai-python
 
