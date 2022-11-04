import csv
import os, glob, argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='./data/dataset/', help='dataset path')
parser.add_argument('--img-size', nargs='+', type=int, default=640, help='train image sizes, always a square')
parser.add_argument('--target', type=str, default='tflite', help='convert to')

opt = parser.parse_args()

if opt.target == 'tflite':
    if not os.path.exists(opt.dataset + 'labels/'):
        print("Nothing to convert (nothing at " + opt.dataset + "labels/")
        exit()

    files = glob.glob(opt.dataset + 'labels/*.txt')

    csv_writer = open(opt.dataset + "labels/annotations.csv", "w+")
    csv_writer.write("filename,xmin,xmax,ymin,ymax,class\n")

    for file in files:
        contents = open(file, 'r').readlines()
        boxes = [[line.split()[0]] + [float(val) for val in line.split()[1:]] for line in contents]

        print(file, contents, boxes)

        for box in boxes:
            box_str = f"{box[0]} {int((box[1] - 0.5*box[3]) * opt.img_size)} {int((box[2] - 0.5*box[4]) * opt.img_size)} {int((box[1] + box[3]) * opt.img_size)} {int((box[2] + box[4]) * opt.img_size)}\n"

            print(box_str)
            csv_writer.write(box_str)
        # csv_writer.write(
        #     f"{values[0]} {values[1]} {values[1] - 2*values[3]} {values[2] - 2*values[4]} {values[1] + values[3]} {values[2] + values[4]}"
        # )

# TODO: tflite -> yolo/opencv 
# has to group together the same [frame number].jpg into one list to write into one file for one image
# elif opt.target == 'opencv':
#     csv_lines = open(opt.dataset + "labels/annotations.csv", "r").readlines()

#     for line in range(len(csv_lines)):
#         open(opt.dataset + 'labels/' + csv_lines[line] + '.txt', 'w').write


