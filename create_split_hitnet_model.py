import onnx_graphsurgeon as gs
import numpy as np
import onnx

from model_conversion_utils import split_model, replace_input

def split_hitnet(model_path):

	onnx_model = onnx.load(model_path)

	input_names = ["input"]
	intermediate_output_names = ["fe_shared/conv_up_2_0/BiasAdd;level0_1/shared/prop0/prop/resblock1/conv2/Conv2D;fe_shared/conv_up_2_0/Conv2D;fe_shared/conv_up_2_0/bias1__259:0",
								 "fe_shared/LeakyRelu_22",
								 "fe_shared/LeakyRelu_18"]
	output_names = ["reference_output_disparity"] # Ignore the secib=

	# Split model into two
	subgraph1 = split_model(onnx_model, input_names, intermediate_output_names)
	onnx.save(gs.export_onnx(subgraph1), "models/subgraph1.onnx")

	subgraph2 = split_model(onnx_model, intermediate_output_names, output_names)
	onnx.save(gs.export_onnx(subgraph2), "models/subgraph2.onnx")

	# Replace the input to have two inputs
	graph = gs.import_onnx(onnx.load("models/subgraph1.onnx"))

	left_rect_img = gs.Variable(name="left_rect", dtype=np.float32, shape=(1, 3, 240, 320))
	right_rect_img = gs.Variable(name="right_rect", dtype=np.float32, shape=(1, 3, 240, 320))
	graph.replace_input([left_rect_img, right_rect_img], "input")

	onnx.save(gs.export_onnx(graph), "models/subgraph1_mod.onnx")

if __name__ == '__main__':
	
	model_path = "models/flyingthings_finalpass_xl/saved_model_240x320/model_float32_opt.onnx"
	split_hitnet(model_path)
