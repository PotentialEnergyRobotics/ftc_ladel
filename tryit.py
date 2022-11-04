import tensorflow as tf
import cv2 as cv
import logging
import os
from natsort import natsorted
import time
logging.basicConfig(level=logging.INFO, filename='tryit.txt', filemode='w', format="%(message)s")
log = logging.getLogger('tryit')

model_path="mechjeb_lite0.tflite"
log.info(f"tf={tf.__version__}, model={model_path}")

#log.info("Load lables")
#Load the label map
with open("labelmap_mechjeb.txt", 'r') as f:
    labels = [line.strip() for line in f.readlines()]

log.info("Load TFLite model and allocate tensors")

interpreter = tf.lite.Interpreter(model_path=model_path, num_threads=8)
# Get input and output tensors.
log.info("Get input and output tensors.")
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
# input details
log.info("----- input details:")
log.info(input_details)
# output details
log.info("----- output details:")
log.info(output_details)
log.info("allocate tensors")
interpreter.allocate_tensors()
log.info("---------- iterate images")
images = natsorted(os.listdir('data\\dataset\\images'))
start_time = time.time()
idx = 0
for file in images:
    idx += 1
    if idx % 10 == 0:
        seconds = time.time() - start_time
        log.info(f"-- 10 images took {seconds:.1f}s to process")
        start_time = time.time()
    log.info(f"---- image={file}")
    # read and resize the image
    img = cv.imread(r"data\\dataset\\images\\{}".format(file))
    img = cv.resize(img, (320, 320))
    #log.info(f"set_tensor")
    
    interpreter.set_tensor(input_details[0]['index'], [img])
    #log.info(f"invoke")
    interpreter.invoke()
    #log.info("Retrieve detection results")
    boxes = interpreter.get_tensor(output_details[0]['index'])[0] # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[1]['index'])[0] # Class index of detected objects
    scores = interpreter.get_tensor(output_details[2]['index'])[0] # Confidence of detected objects
    #log.info(f"scores={scores}")
    if not hasattr(scores, "__len__"):
        scores = [scores]
    #log.info(f"scores2={scores}")
    for i in range(len(scores)):
        if ((scores[i] > 0.5) and (scores[i] <= 1.0)):
            object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
            label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
            log.info(f"\t{label}")

    # output_data = interpreter.get_tensor(output_details[0]['index'])
    # log.info(f"output_data {output_data}")
    # if (output_data[0][0] > 200):
    #     log.info("file {} -> zero".format(file.stem))
    # elif (output_data[0][1] > 200):
    #     log.info("file {} -> one".format(file.stem))
    # elif (output_data[0][2] > 200):
    #     log.info("file {} -> two".format(file.stem))
    # elif (output_data[0][3] > 200):
    #     log.info("file {} -> three".format(file.stem))
    # else:
    #     log.info("file {} -> unknown".format(file.stem))