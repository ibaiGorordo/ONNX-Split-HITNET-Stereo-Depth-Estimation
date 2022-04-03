from os.path import exists
import cv2
import numpy as np
from imread_from_url import imread_from_url

from hitnet_split import  HitNet_Split1, HitNet_Split2, CameraConfig
from create_split_hitnet_model import split_hitnet

hitnet1_path = "models/subgraph1_mod.onnx"
hitnet2_path = "models/subgraph2.onnx"
original_hitnet_path = "models/flyingthings_finalpass_xl/saved_model_240x320/model_float32_opt.onnx"

# Split the model if it has not been split before
if not exists(hitnet1_path):
	split_hitnet(original_hitnet_path)


# Initialize models
hitnet1 = HitNet_Split1(hitnet1_path)
hitnet2 = HitNet_Split2(hitnet2_path)

# Load images
left_img = imread_from_url("https://vision.middlebury.edu/stereo/data/scenes2003/newdata/cones/im2.png")
right_img = imread_from_url("https://vision.middlebury.edu/stereo/data/scenes2003/newdata/cones/im6.png")

intermediate_outputs = hitnet1(left_img, right_img)
disparity_map = hitnet2(intermediate_outputs)

color_disparity = hitnet2.draw_disparity(left_img.shape[1::-1])
combined_image = np.hstack((left_img, color_disparity))

cv2.imwrite("out.jpg", combined_image)

cv2.namedWindow("Estimated disparity", cv2.WINDOW_NORMAL)	
cv2.imshow("Estimated disparity", combined_image)
cv2.waitKey(0)

cv2.destroyAllWindows()




