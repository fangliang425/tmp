import os
import numpy as np
import cv2
import glob

from openvino.inference_engine import IENetwork, IECore
from PIL import Image

class OpenVINODetector:

    def __init__(self, model_path, device="CPU", classes=None):
        # Plugin initialization for specified device and load extensions library if specified
        print("Creating Inference Engine")
        ie = IECore()
        # Read IR
        print(f"Loading network files from folder {model_path}")
        # Decode model files if needed
        tmpfiles = []
        model_xml = glob.glob(model_path + os.sep + "*.xml")[0]
        model_bin = glob.glob(model_path + os.sep + "*.bin")[0]

        net = IENetwork(model=model_xml, weights=model_bin)
        # Remove decoded model files
        for name in tmpfiles:
            os.unlink(name)
        print("Preparing input blobs")

        if "image_tensor" in net.inputs:  # required for rcnn models
            self.input_blob = "image_tensor"
        else:
            self.input_blob = next(iter(net.inputs))

        self.output_blob = next(iter(net.outputs))
        net.batch_size = 1

        # Loading model to the plugin
        print("Loading model to the plugin")
        self.exec_net = ie.load_network(network=net, device_name=device)

        # Get network expected input size
        n, c, h, w = net.inputs[self.input_blob].shape
        self.width = w
        self.height = h

        self.classes = classes

        print("Initialization finished.")
        print(f"Input shape is wxh = {w}x{h}.")


    def detect(self, image):

        input_height, input_width, _ = image.shape
        # Run detection network on one image.
        image = cv2.resize(image, (self.width, self.height))
        image = image.transpose((2, 0, 1))  # Change data layout from HWC to CHW
        net_input = image[np.newaxis, ...]

        # Pass data through the net
        result = self.exec_net.infer(inputs={self.input_blob: net_input})

        return result



# image path
image = np.array(Image.open("<path>/image1.jpg"))
image_cpy = image.copy()
height, width, channel = image.shape

#openvino path
model_openvino_path = "<path>/EfficientDet_Openvino"
detector_openvino = OpenVINODetector(model_openvino_path)
results = detector_openvino.detect(image)
detections = results["DetectionOutput"][0][0]

image_ids = detections[:, 0]
labels = detections[:, 1]
confs = detections[:, 2]

x_mins = detections[:, 3] * width
y_mins = detections[:, 4] * height
x_maxs = detections[:, 5] * width
y_maxs = detections[:, 6] * height

idxs = confs > 0.5
image_ids = image_ids[idxs]
labels = labels[idxs].astype(np.int)
confs = confs[idxs]
x_mins = x_mins[idxs]
y_mins = y_mins[idxs]
x_maxs = x_maxs[idxs]
y_maxs = y_maxs[idxs]
boxes = np.vstack([y_mins, x_mins, y_maxs, x_maxs]).transpose().astype(np.int)

for label, box, conf in zip(labels, boxes, confs):
    y_min, x_min, y_max, x_max = box

    cv2.rectangle(image_cpy, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)  # bbox
    cv2.putText(image_cpy, str(label), (x_min,  y_max + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)  # class
    cv2.putText(image_cpy, str(round(conf, 2)), (x_min + 50, y_max + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

image_cpy = cv2.cvtColor(image_cpy, cv2.COLOR_RGB2BGR)


cv2.imshow("image", image_cpy)
cv2.waitKey(0)
cv2.destroyAllWindows()
