import os
import numpy as np
import cv2

import tensorflow as tf
from PIL import Image



gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])


# image path
image = np.array(Image.open("<path>/image1.jpg"))
image_cpy = image.copy()
height, width, channel = image.shape
image = cv2.resize(image, (512, 512))
input_tensor = tf.convert_to_tensor(image)
input_tensor = input_tensor[tf.newaxis, ...]

# tensorflow saved model path
model_tf_path = "<path>/efficientdet_d0_coco17_tpu-32/saved_model"
detect_fn = tf.saved_model.load(model_tf_path)
detections_tf = detect_fn(input_tensor)

num_detections = int(detections_tf.pop('num_detections'))
detections_tf = {key: value[0, :num_detections].numpy() for key, value in detections_tf.items()}
detections_tf['num_detections'] = num_detections
detections_tf['detection_classes'] = detections_tf['detection_classes'].astype(np.int64)

boxes = detections_tf['detection_boxes']
boxes[:, 0] = boxes[:, 0] * height
boxes[:, 1] = boxes[:, 1] * width
boxes[:, 2] = boxes[:, 2] * height
boxes[:, 3] = boxes[:, 3] * width
boxes = boxes.astype(np.int)

labels = detections_tf['detection_classes']
confs = detections_tf['detection_scores']

idxs = confs > 0.5
labels = labels[idxs]
confs = confs[idxs]
boxes = boxes[idxs]

for label, box, conf in zip(labels, boxes, confs):

    y_min, x_min, y_max, x_max = box

    cv2.rectangle(image_cpy, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)  # bbox
    cv2.putText(image_cpy, str(label), (x_min,  y_max + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)  # class
    cv2.putText(image_cpy, str(round(conf, 2)), (x_min + 50,  y_max + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

image_cpy = cv2.cvtColor(image_cpy, cv2.COLOR_RGB2BGR)
cv2.imshow("image", image_cpy)
cv2.waitKey(0)
cv2.destroyAllWindows()


