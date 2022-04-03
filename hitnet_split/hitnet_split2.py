from enum import Enum
from dataclasses import dataclass

import cv2
import numpy as np
import onnxruntime

@dataclass
class CameraConfig:
    baseline: float
    f: float

DEFAULT_CONFIG = CameraConfig(0.546, 120) # rough estimate from the original calibration

class HitNet_Split2():

	def __init__(self, model_path, camera_config=DEFAULT_CONFIG, max_dist=10):

		# Initialize model
		self.model = self.initialize_model(model_path, camera_config, max_dist)

	def __call__(self, inputs):

		return self.update(inputs)

	def initialize_model(self, model_path, camera_config=DEFAULT_CONFIG, max_dist=10):
		
		self.camera_config = camera_config
		self.max_dist = max_dist

		# Initialize model session
		self.session = onnxruntime.InferenceSession(model_path, providers=['CUDAExecutionProvider',
																		   'CPUExecutionProvider'])

		# Get model info
		self.get_input_details()
		self.get_output_details()

	def update(self, inputs):

		output = self.inference(inputs)
		self.disparity_map = self.process_output(output)

		# Estimate depth map from the disparity
		self.depth_map = self.get_depth_from_disparity(self.disparity_map, self.camera_config)

		return self.disparity_map

	def inference(self, inputs):

		inputs = {name:value for name, value in zip(self.input_names, inputs)}
		return self.session.run(self.output_names, inputs)

	@staticmethod
	def process_output(output): 

		return np.squeeze(output)

	@staticmethod
	def get_depth_from_disparity(disparity_map, camera_config):

		return camera_config.f*camera_config.baseline/disparity_map

	def draw_disparity(self, out_shape):

		disparity_map =  cv2.resize(self.disparity_map,  out_shape)
		norm_disparity_map = 255*((disparity_map-np.min(disparity_map))/
								  (np.max(disparity_map)-np.min(disparity_map)))

		return cv2.applyColorMap(cv2.convertScaleAbs(norm_disparity_map,1), cv2.COLORMAP_MAGMA)

	def draw_depth(self, out_shape):

		norm_depth_map = 255*(1-self.depth_map/self.max_dist)
		norm_depth_map[norm_depth_map < 0] = 0
		norm_depth_map[norm_depth_map >= 255] = 0

		norm_depth_map =  cv2.resize(norm_depth_map, out_shape)

		return cv2.applyColorMap(cv2.convertScaleAbs(norm_depth_map,1), cv2.COLORMAP_MAGMA)	

	def get_input_details(self):

		model_inputs = self.session.get_inputs()
		self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]

	def get_output_details(self):

		model_outputs = self.session.get_outputs()
		self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]
	
if __name__ == '__main__':

	import pickle
	from imread_from_url import imread_from_url
	
	model_path =  "../models/subgraph2.onnx"

	# Load image for plotting only
	left_img = imread_from_url("https://vision.middlebury.edu/stereo/data/scenes2003/newdata/cones/im2.png")

	# Initialize model
	hitnet2 = HitNet_Split2(model_path)

	# Load input
	with open('int_outputs.pickle', 'rb') as f:
		intermediate_outputs = pickle.load(f)

	# Estimate the depth
	disparity_map = hitnet2(intermediate_outputs)

	color_disparity = hitnet2.draw_disparity(left_img.shape[1::-1])
	combined_image = np.hstack((left_img, color_disparity))

	cv2.imwrite("out.jpg", combined_image)

	cv2.namedWindow("Estimated disparity", cv2.WINDOW_NORMAL)	
	cv2.imshow("Estimated disparity", combined_image)
	cv2.waitKey(0)

	cv2.destroyAllWindows()






