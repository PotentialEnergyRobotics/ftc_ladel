"""
Usage:
    # Create train data:
    python generate_tfrecord.py --csv_input=data/dataset/labels/annotations.csv --output_path=data/tfrecords/train.record --image_dir=data/dataset/images
     [or /train and (annotations)_train.csv]
    # Create test data:
    python generate_tfrecord.py --csv_input=data/dataset/labels/annotations_test.csv --output_path=data/tfrecords/test.record --image_dir=data/dataset/images/test
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import io
import pandas as pd
import tensorflow as tf

from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict

def load_pbtxt(pbtxt_path):
	labels = []
	if os.path.isfile(pbtxt_path):
		with open(pbtxt_path, 'r') as lm:
			labels = [line.split("'")[1] for line in lm.readlines() if "'" in line] # kind of crappy but should work :)
	return labels

labels = load_pbtxt("data/labelmap.pbtxt")

flags = tf.compat.v1.flags
flags.DEFINE_string("csv_input", "", "data/dataset/labels/annotations.csv")
flags.DEFINE_string("output_path", "", "data/tfrecords/train.record")
flags.DEFINE_string("image_dir", "", "data/dataset/images")
FLAGS = flags.FLAGS

# wooow such code I am good at coding 
if not os.path.exists(FLAGS.output_path):
    os.mkdir(FLAGS.image_dir.split('dataset')[0] + 'tfrecord')

# reference https://blog.roboflow.com/create-tfrecord/
def class_text_to_int(row_label):
	return labels.index(row_label) + 1

def split(df, group):
    data = namedtuple("data", ["filename", "object"])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]

def create_tf_example(group, path):
    with tf.compat.v1.gfile.GFile(os.path.join(path, "{}".format(group.filename)), "rb") as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode("utf8")
    image_format = b"jpg"
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row["xmin"] / width)
        xmaxs.append(row["xmax"] / width)
        ymins.append(row["ymin"] / height)
        ymaxs.append(row["ymax"] / height)
        classes_text.append(row["class"].encode("utf8"))
        classes.append(class_text_to_int(row["class"]))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        "image/height": dataset_util.int64_feature(height),
        "image/width": dataset_util.int64_feature(width),
        "image/filename": dataset_util.bytes_feature(filename),
        "image/source_id": dataset_util.bytes_feature(filename.split(image_format)[0]),
        "image/encoded": dataset_util.bytes_feature(encoded_jpg),
        "image/format": dataset_util.bytes_feature(image_format),
        "image/object/bbox/xmin": dataset_util.float_list_feature(xmins),
        "image/object/bbox/xmax": dataset_util.float_list_feature(xmaxs),
        "image/object/bbox/ymin": dataset_util.float_list_feature(ymins),
        "image/object/bbox/ymax": dataset_util.float_list_feature(ymaxs),
        "image/object/class/text": dataset_util.bytes_list_feature(classes_text),
        "image/object/class/label": dataset_util.int64_list_feature(classes),
    }))
    return tf_example

def main(_):
    writer = tf.compat.v1.python_io.TFRecordWriter(FLAGS.output_path)
    path = os.path.join(FLAGS.image_dir)
    examples = pd.read_csv(FLAGS.csv_input)
    grouped = split(examples, "filename")
    for group in grouped:
        tf_example = create_tf_example(group, path)
        writer.write(tf_example.SerializeToString())

    writer.close()
    output_path = os.path.join(os.getcwd(), FLAGS.output_path)
    print("Successfully created the TFRecords: {}".format(output_path))

if __name__ == "__main__":
	tf.compat.v1.app.run()
