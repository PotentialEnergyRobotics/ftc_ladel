
# Ladel Pipeline

![The procrastination banner](./images/ladelbanner.png)

Ladel is a TFLite+YoloV7 enabled labeling and training pipeline built for [First Tech Challenge](<https://www.firstinspires.org/robotics/ftc>)/TF 2.5, usable anywhere.

- [Ladel Pipeline](#ladel-pipeline)
  - [Note](#note)
  - [Installation](#installation)
  - [Labeling](#labeling)
    - [Labeling keys](#labeling-keys)
  - [Training](#training)
  - [Other Files](#other-files)
  - [Implementation](#implementation)
  - [Bugs](#bugs)
  - [Contributing](#contributing)
    - [Some potential areas for improvment](#some-potential-areas-for-improvment)
  - [License](#license)

Ladel is also a misspell of label. Creative, I know.

## Note

Most files have a `--help` option for options. Others will simply have information within the file. View [SUFFERING.md](./SUFFERING.md) for the story and why this exists.

## Installation

```bash
git clone https://github.com/PotentialEnergyRobotics/ladel
conda env create --file tf_2_5.yml
conda activate tf_2_5
pip install -r requirements.txt
```

## Labeling

AKA the best part

`python ladel.py`

![Gif of Label Studio annotating different types of data](./images/ladelpy.GIF)

### Labeling keys

- 'A' key goes back a frame
- 'D' key goes forward a frame
- 'P' key pauses or plays the video
- 'i' key adds new bounding boxes
- 's' key deletes all bounding boxes
- 'lmb' left mouse deletes a specific bounding box clicked within
- You can purge all the files on each run
- You can start from the last frame saved on each run (to use more than one video)
- Input labels in the console (cannot be a number because of the way tfrecords work)

## Training

- Generate a file containing data with format used by train
  - `python generate_tfrecord.py --csv_input=data/labels/annotations.csv --output_path=data/dataset/tfrecords/train.record --image_dir=data/dataset/images`
- One-shot make your model file! If you run out of memory, decrease the batch size
  - `python train.py`

## Other Files

- splitter.py
  - used for splitting YOLOV7 data into train and test datasets
- converter.py
  - can currently convert from YOLOV7 format to tflite format
- tryit.py
  - use the webcam to try out tflite models (don't expect a decent FPS)

## Implementation

For FTC folks, mostly.

- You need a [Vuforia](https://developer.vuforia.com/) key.
- Plug into your robot controller,  put the generated tflite file somewhere, and copy the path.
- In Android Studio, copy `FtcRobotController/src/main/java/org/firstinspires/ftc/robotcontroller/external/samples/ConceptTensorFlowObjectDetectionWebcam.java` into your team's code folder
- Remove `@disabled`.
- Change the `LABELS` to your labels.
- Update `TFOD_MODEL_FILE` model to your file path.
- Uncomment `tfod.loadModelFromFile` and comment load from `tfod.loadModelFromAsset`.

## Bugs

Use issues tab to report it, and we'll get to it ASAP.

## Contributing

[Fork it, change it, pull request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork). Or make a feature request in issues tab. I'll get to it ASAP.

### Some potential areas for improvment

- [ ] Improved/advanced labeler
  - [ ] Frame skipping
  - [ ] Label selection with scroll wheel
  - [ ] Other ideas...
- [ ] Write additional help messages and documentation (where needed, such as ladel behavior)
- [ ] Better code, currently practicing anarchy system
- [ ] Graphic design help
- [ ] Better YOLOV7 support

For now this is firmly in the pre-alpha state, if you could call it that

## License

Adding a license is difficult as a significant portion of the code is copied line-for-line, and I'm not a lawyer. I'd say do whatever you want, and the original authors probably won't care but if you really need to be sure just use google.com to search some portion of the code and check whatever comes up for license.
