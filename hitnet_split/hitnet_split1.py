import cv2
import numpy as np
import onnxruntime

class HitNet_Split1():

	def __init__(self, model_path):

		# Initialize model
		self.model = self.initialize_model(model_path)

	def __call__(self, left_img, right_img):

		return self.update(left_img, right_img)

	def initialize_model(self, model_path):

		# Initialize model session
		self.session = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])

		# Get model info
		self.get_input_details()
		self.get_output_details()

	def update(self, left_img, right_img):

		left_tensor = self.prepare_input(left_img)
		right_tensor = self.prepare_input(right_img)
		return self.inference(left_tensor, right_tensor)

	def prepare_input(self, img):

		input_img = cv2.resize(img, (self.input_width, self.input_height))
		input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB) 

		input_img = input_img.astype(np.float32)/ 255.0
		input_img = input_img.transpose(2, 0, 1)

		return input_img[np.newaxis,:,:,:]   

	def inference(self, left_tensor, right_tensor):

		return self.session.run(self.output_names, {self.input_names[0]: left_tensor,
													self.input_names[1]: right_tensor})

	def get_input_details(self):

		model_inputs = self.session.get_inputs()
		self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]

		self.input_shape = model_inputs[0].shape
		self.input_height = self.input_shape[2]
		self.input_width = self.input_shape[3]

	def get_output_details(self):

		model_outputs = self.session.get_outputs()
		self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]

		self.output_shape = model_outputs[0].shape
	
if __name__ == '__main__':

	from imread_from_url import imread_from_url
	import pickle

	model_path = "../models/subgraph1_mod.onnx"

	# Initialize model
	hitnet1 = HitNet_Split1(model_path)

	# Load images
	left_img = imread_from_url("https://vision.middlebury.edu/stereo/data/scenes2003/newdata/cones/im2.png")
	right_img = imread_from_url("https://vision.middlebury.edu/stereo/data/scenes2003/newdata/cones/im6.png")

	intermediate_outputs = hitnet1(left_img, right_img)

	with open('int_outputs.pickle', 'wb') as f:
		pickle.dump(intermediate_outputs, f)







