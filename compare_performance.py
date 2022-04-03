from os.path import exists
import cv2
import numpy as np
import time
from imread_from_url import imread_from_url

from hitnet_split import  HitNet_Split1, HitNet_Split2
from hitnet import HitNet, ModelType
from create_split_hitnet_model import split_hitnet

hitnet1_path = "models/subgraph1_mod.onnx"
hitnet2_path = "models/subgraph2.onnx"
original_hitnet_path = "models/flyingthings_finalpass_xl/saved_model_240x320/model_float32_opt.onnx"

# Split the model if it has not been split before
if not exists(hitnet1_path):
	split_hitnet(original_hitnet_path)

# Initialize models
hitnet = HitNet(original_hitnet_path, ModelType.flyingthings)
hitnet1 = HitNet_Split1(hitnet1_path)
hitnet2 = HitNet_Split2(hitnet2_path)

# Create inputs
left_input = np.ones((480,640,3),dtype=np.float32)
right_input = np.ones((480,640,3),dtype=np.float32)

## Split version
num_tests = 10
elapsed_times_split = []
for i in range(num_tests + 1):
	start_time = time.monotonic()
	intermediate_outputs = hitnet1(left_input, right_input)
	disparity_map = hitnet2(intermediate_outputs)
	elapsed_times_split.append(time.monotonic() - start_time)

# Calculate the average inference time skipping the first one
avg_inf_time_split = np.array(elapsed_times_split[1:]).mean()*1000

## Original version
num_tests = 10
elapsed_times_orig = []
for i in range(num_tests + 1):
	start_time = time.monotonic()
	disparity_map = hitnet(left_input, right_input)
	elapsed_times_orig.append(time.monotonic() - start_time)

# Calculate the average inference time skipping the first one
avg_inf_time_orig = np.array(elapsed_times_orig[1:]).mean()*1000


print(f"Avg. Inference Time - Original: {avg_inf_time_orig} ms, Split: {avg_inf_time_split} ms")






