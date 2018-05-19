#! /usr/bin/env python
"""Run a YOLO_v2 style detection model on test images."""
import argparse
import colorsys
import os
import random

import numpy as np
from keras import backend as K
from keras.models import load_model
import cv2
import cv2.ocl as ocl
import common
from pascal_voc_parser import get_data
from common_measure_method import get_map
from common_measure_method import output_ap
from common_measure_method import draw_measure_curve
from yad2k.models.keras_yolo import yolo_eval, yolo_head

parser = argparse.ArgumentParser(
    description='Run a YOLO_v2 style detection model on test images..')
parser.add_argument(
    'model_path',
    help='path to h5 model file containing body'
    'of a YOLO_v2 model')
parser.add_argument(
    '-a',
    '--anchors_path',
    help='path to anchors file, defaults to yolo_anchors.txt',
    default='model_data/yolo_anchors.txt')
parser.add_argument(
    '-c',
    '--classes_path',
    help='path to classes file, defaults to coco_classes.txt',
    default='model_data/coco_classes.txt')
parser.add_argument(
    '-s',
    '--score_threshold',
    type=float,
    help='threshold for bounding box scores, default .3',
    default=.3)
parser.add_argument(
    '-iou',
    '--iou_threshold',
    type=float,
    help='threshold for non max suppression IOU, default .5',
    default=.5)

parser.add_argument(
    '-p',
    '--test_path',
    help='image_test_path',)

def _main(args):
    model_path = os.path.expanduser(args.model_path)
    assert model_path.endswith('.h5'), 'Keras model must be a .h5 file.'
    anchors_path = os.path.expanduser(args.anchors_path)
    classes_path = os.path.expanduser(args.classes_path)

    sess = K.get_session()

    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]

    with open(anchors_path) as f:
        anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        anchors = np.array(anchors).reshape(-1, 2)

    yolo_model = load_model(model_path)

    # Verify model, anchors, and classes are compatible
    num_classes = len(class_names)
    num_anchors = len(anchors)
    model_output_channels = yolo_model.layers[-1].output_shape[-1]
    assert model_output_channels == num_anchors * (num_classes + 5), \
        'Mismatch between model and given anchor and class sizes. ' \
        'Specify matching anchors and classes with --anchors_path and ' \
        '--classes_path flags.'
    print('{} model, anchors, and classes loaded.'.format(model_path))

    # Check if model is fully convolutional, assuming channel last order.
    model_image_size = yolo_model.layers[0].input_shape[1:3]
    is_fixed_size = model_image_size != (None, None)

    # Generate colors for drawing bounding boxes.
    hsv_tuples = [(x / len(class_names), 1., 1.)
                  for x in range(len(class_names))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(
        map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
            colors))
    random.seed(10101)  # Fixed seed for consistent colors across runs.
    random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    random.seed(None)  # Reset seed to default.

    # Generate output tensor targets for filtered bounding boxes.
    yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))
    input_image_shape = K.placeholder(shape=(2, ))
    boxes, scores, classes = yolo_eval(
        yolo_outputs,
        input_image_shape,
        score_threshold=args.score_threshold,
        iou_threshold=args.iou_threshold)

    test_path = os.path.expanduser(args.test_path)
    all_imgs, _, _ = get_data(test_path)
    test_imgs = [s for s in all_imgs if s['imageset'] == 'test']

    for idx, img_data in enumerate(test_imgs):
        try:
            print('{}/{}'.format(idx, len(test_imgs)))
            filepath = img_data['filepath']
            print(f"file = {filepath}")

            image = cv2.imread(filepath)

            # Due to skip connection + max pooling in YOLO_v2, inputs must have
            # width and height as multiples of 32.
            resized_image = cv2.resize(image, tuple(reversed(model_image_size)))
            resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
            image_data = np.array(resized_image, dtype='float32')

            image_size = image.shape

            image_data /= 255.
            image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

            out_boxes, out_scores, out_classes = sess.run(
                [boxes, scores, classes],
                feed_dict={
                    yolo_model.input: image_data,
                    input_image_shape: [int(image_size[0]), int(image_size[1])],
                    K.learning_phase(): 0
                })

            for i, c in reversed(list(enumerate(out_classes))):
                predicted_class = class_names[c]
                box = out_boxes[i]
                score = out_scores[i]

                textLabel = '{} {:.2f}'.format(predicted_class, score)

                top, left, bottom, right = box
                top = max(0, np.floor(top + 0.5).astype('int32'))
                left = max(0, np.floor(left + 0.5).astype('int32'))
                bottom = min(image_size[0], np.floor(bottom + 0.5).astype('int32'))
                right = min(image_size[1], np.floor(right + 0.5).astype('int32'))
                print(textLabel, (left, top), (right, bottom))


        except Exception as e:
            print(f"Some exception occur!\n\r {e} \n we don't care")

    sess.close()

if __name__ == '__main__':
    ocl.setUseOpenCL(True)
    _main(parser.parse_args())
