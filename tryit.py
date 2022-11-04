import os, argparse, logging, time
logging.basicConfig(level=logging.INFO, filename='tryit.txt', filemode='w', format="%(message)s")
log = logging.getLogger('tryit')

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='./data/model_out/model.tflite', help='model path')
parser.add_argument('--labelmap', type=str, default='./data/labelmap.pbtxt', help='labelmap path')
parser.add_argument('--img-size', nargs='+', type=int, default=640, help='image sizes, always a square')
parser.add_argument('--min-confidence', type=float, default=0.1, help='at what point should tryit spit out a label?')

opt = parser.parse_args()

import tensorflow as tf
import cv2 as cv
from natsort import natsorted

log.info(f"tf={tf.__version__}, model={opt.model}")

# Load labelmap
def load_pbtxt(pbtxt_path):
	labels = []
	if os.path.isfile(pbtxt_path):
		with open(pbtxt_path, 'r') as lm:
			labels = [line.split("'")[1] for line in lm.readlines() if "'" in line] # kind of crappy but should work :)
	return labels

labels = load_pbtxt(opt.labelmap)
log.info(f"labels: {labels}")

log.info("Load TFLite model and allocate tensors")

interpreter = tf.lite.Interpreter(model_path=opt.model, num_threads=8)

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
    img = cv.resize(img, (opt.img_size, opt.img_size))
    
    interpreter.set_tensor(input_details[0]['index'], [img])
    interpreter.invoke()
    boxes = interpreter.get_tensor(output_details[0]['index'])[0] # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[1]['index'])[0] # Class index of detected objects
    scores = interpreter.get_tensor(output_details[2]['index'])[0] # Confidence of detected objects
    log.info(f"scores={scores}")
    if not hasattr(scores, "__len__"):
        scores = [scores]
    for i in range(len(scores)):
        if ((scores[i] > opt.min_confidence) and (scores[i] <= 1.0)):
            object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
            label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
            log.info(f"\t{label}")

print("Check ./tryit.txt")