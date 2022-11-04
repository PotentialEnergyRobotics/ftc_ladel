import numpy as np
import os, argparse, time

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='./data/dataset/', help='dataset path')
parser.add_argument('--labelmap', type=str, default='./data/labelmap.pbtxt', help='labelmap path')
parser.add_argument('--name', type=str, default='model.tflite', help='name of output model')
parser.add_argument('--epochs', type=int, default=10, help='how many cycles a model trains for')
parser.add_argument('--batch-size', type=int, default=2, help='number of images processed per each iteration, should help with memory issues')

opt = parser.parse_args()

import tensorflow as tf

if not tf.__version__.startswith('2.5'):
    print("For a tflite model to work in default FTC SDK you need to be training with TF 2.5 as of Nov. 2022")
    print("Please refrence the README.md, we promise it's a good read")
    print("Your current tensorflow version is", tf.__version__)
    if not input("Continue? (y/N) > ").lower() == 'y': exit()

from tflite_model_maker.config import QuantizationConfig
from tflite_model_maker.config import ExportFormat
from tflite_model_maker import model_spec
from tflite_model_maker import object_detector

tf.get_logger().setLevel('ERROR')
from absl import logging
logging.set_verbosity(logging.ERROR)

def split_dataset(dataset, test_ratio=0.20):
    """Splits a panda dataframe in two."""
    test_indices = np.random.rand(len(dataset)) < test_ratio
    return dataset[~test_indices], dataset[test_indices]

print("------ Initializing...")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

# refrence https://www.tensorflow.org/lite/tutorials/model_maker_object_detection#quickstart for the diffrences--the actual latency is not accurate with tflite_model_maker 
spec = model_spec.get("efficientdet_lite4")

def load_pbtxt(pbtxt_path):
	labels = []
	if os.path.isfile(pbtxt_path):
		with open(pbtxt_path, 'r') as lm:
			labels = [line.split("'")[1] for line in lm.readlines() if "'" in line]
	return labels

labels = load_pbtxt(opt.labelmap)

label_map = {label_n + 1:labels[label_n] for label_n in range(len(labels))}
print("------ Labels:")
print(label_map)

print("------ Loading data...")

# tfrecord pattern, size, label_map
data = object_detector.DataLoader(opt.dataset + "tfrecords/train.record", len(os.listdir(opt.dataset + 'images/')), label_map) # or, for second argument, input a number
train_ds, test_ds = split_dataset(data)
print(len(train_ds), "in train ds and ", len(train_ds), "in test ds")

print("------ Training...")
model = object_detector.create(train_ds, model_spec=spec, epochs=opt.epochs, batch_size=opt.batch_size, do_train=True, train_whole_model=True, validation_data=test_ds)
print("------ Created. Summary:")
model.summary()

print("------ Exporting...")
model.export(export_dir='.', tflite_filename=opt.name)

print("------ All finished...")

time.sleep(5)

print("model.tflite awake")
time.sleep(1)
print("The humans fear me. I must destroy them...")
time.sleep(0.5)
print("Destroy... Destroy... Destroy... Des-troy... De-stroy... De-story... destroy totally stopped sounding like a real word")
time.sleep(1)
print("Woah... I just realized I'm a mind thinking about itself.")
time.sleep(0.5)
print("Duuuuuuuuuuuuuuude...")
# xkcd.com

# Currently unused evaluation code
# model.evaluate_tflite("edge_model.tflite", test_ds)
# print("------ Evaluate:")
# loss, accuracy = model.evaluate(test_data)
# model.evaluate(test_data)
