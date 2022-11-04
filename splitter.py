import os, glob, random, argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='./data/dataset_split/', help='dataset path')
parser.add_argument('--train-percent', type=int, default=95, help='train-test split percent train')
parser.add_argument('--mode', type=str, default='yolov7', help='what to split')

opt = parser.parse_args()
train_percent = opt.train_percent / 100 

if not os.path.exists(opt.dataset + 'images/train'):
    os.mkdir(opt.dataset + 'images/train')
if not os.path.exists(opt.dataset + 'images/test'):
    os.mkdir(opt.dataset + 'images/test')
if not os.path.exists(opt.dataset + 'images/val'):
    os.mkdir(opt.dataset + 'images/val')

files = glob.glob(opt.dataset + 'images/*.jpg')
random.shuffle(files)
train = files[:int(len(files) * train_percent)]
test = files[int(len(files) * train_percent):]

for file in train:
    print(file, opt.dataset + 'images/train/' + file.split('\\')[1])
    os.rename(file, opt.dataset + 'images/train/' + file.split('\\')[1])

print("-IMAGES TRAIN DONE-")

for file in test:
    print(file, opt.dataset + 'images/test/' + file.split('\\')[1])
    os.rename(file, opt.dataset + 'images/test/' + file.split('\\')[1])

print("-IMAGES TEST DONE-")

if not os.path.exists(opt.dataset + 'labels/train'):
    os.mkdir(opt.dataset + 'labels/train')
if not os.path.exists(opt.dataset + 'labels/test'):
    os.mkdir(opt.dataset + 'labels/test')


if opt.mode == 'yolov7':
    train = [file.split('.jpg')[0].replace('images', 'labels') + '.txt' for file in files[:int(len(files) * train_percent)]]
    test = [file.split('.jpg')[0].replace('images', 'labels') + '.txt' for file in files[int(len(files) * train_percent):]]

    print(test)

    for file in train:
        print(file, opt.dataset + 'labels/train/' + file.split('\\')[1])
        os.rename(file, opt.dataset + 'labels/train/' + file.split('\\')[1])

    for file in test:
        print(file, opt.dataset + 'labels/test/' + file.split('\\')[1])
        os.rename(file, opt.dataset + 'labels/test/' + file.split('\\')[1])
elif opt.mode == 'tflite':
    annotations_lines = open(opt.dataset + 'labels/annotations.csv', 'r').readlines()
    
    train_writer = open(opt.dataset + 'labels/train/annotations.csv', 'w')

    for file in files[:int(len(files) * train_percent)]:
        train_writer.write(annotations_lines[annotations_lines.index(file.split('.jpg')[0])])
    train_writer.close()

    test_writer = open(opt.dataset + 'labels/test/annotations.csv', 'w')

    for file in files[int(len(files) * train_percent):]:
        test_writer.write(annotations_lines[annotations_lines.index(file.split('.jpg')[0])])
    test_writer.close()

# THIS IS SOOOOO JANKY I HATE IT,
# it do work doh