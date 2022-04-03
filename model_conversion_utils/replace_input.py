import onnx_graphsurgeon as gs
import numpy as np
import onnx

@gs.Graph.register()
def replace_input(self, new_inputs, old_input_name):

	tensors = self.tensors()

	# Create concat node to combine the new inputs
	concat_node = gs.Node(op="Concat", attrs={"axis": 1}, inputs=new_inputs, outputs=[tensors[old_input_name]])

	# Replace graph inputs
	self.inputs = new_inputs

	# Insert the concat node at the beginning
	self.nodes.insert(0, concat_node)

	# Remove the now-dangling subgraph.
	self.cleanup().toposort()

if __name__ == '__main__':

	graph = gs.import_onnx(onnx.load("subgraph1.onnx"))

	left_rect_img = gs.Variable(name="left_rect", dtype=np.float32, shape=(1, 3, 240, 320))
	right_rect_img = gs.Variable(name="right_rect", dtype=np.float32, shape=(1, 3, 240, 320))

	graph.replace_input([left_rect_img, right_rect_img], "input")

	onnx.save(gs.export_onnx(graph), "replaced.onnx")