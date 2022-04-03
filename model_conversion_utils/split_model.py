import onnx
import numpy as np
import onnx_graphsurgeon as gs

def split_model(onnx_model, input_names, output_names):
	graph = gs.import_onnx(onnx_model)
	tensors = graph.tensors()

	graph.inputs = [tensors[input_name] for input_name in input_names]
	graph.outputs = [tensors[output_name] for output_name in output_names]
	graph.cleanup()

	return graph

if __name__ == '__main__':
	
	onnx_model = onnx.load('model_float32_opt.onnx')

	input_names = ["input"]
	intermediate_output_names = ["fe_shared/conv_up_2_0/BiasAdd;level0_1/shared/prop0/prop/resblock1/conv2/Conv2D;fe_shared/conv_up_2_0/Conv2D;fe_shared/conv_up_2_0/bias1__259:0",
								 "fe_shared/LeakyRelu_22",
								 "fe_shared/LeakyRelu_18"]
	output_names = ["reference_output_disparity"]							 

	subgraph1 = split_model(onnx_model, input_names, intermediate_output_names)
	onnx.save(gs.export_onnx(subgraph1), "subgraph1.onnx")

	subgraph2 = split_model(onnx_model, intermediate_output_names, output_names)
	onnx.save(gs.export_onnx(subgraph2), "subgraph2.onnx")
