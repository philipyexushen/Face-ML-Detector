#! /usr/bin/env python
"""Run a YOLO_v2 style detection model on test images."""
import argparse
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
import pickle
import traceback
import sys

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
    '-iou',
    '--iou_threshold',
    type=float,
    help='threshold for non max suppression IOU, default .5',
    default=.5)

parser.add_argument(
    '-p',
    '--test_path',
    help='image_test_path',)

parser.add_argument(
    "-usd",
    "--use_store_data",
    help="use_store_data",
    default="0")

parser.add_argument(
    "-mAPFile",
    "--measure_mAP_filename",
    help="Location to store measure mAP file",
    default="measure_mAP_yolo.pickle")

def _main(args):
    model_path = os.path.expanduser(args.model_path)
    assert model_path.endswith('.h5'), 'Keras model must be a .h5 file.'
    anchors_path = os.path.expanduser(args.anchors_path)
    classes_path = os.path.expanduser(args.classes_path)

    sess = K.get_session()

    with open(classes_path) as f:
        yolo_class_names = f.readlines()
    yolo_class_names = [c.strip() for c in yolo_class_names]
    print(yolo_class_names)

    with open(anchors_path) as f:
        anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        anchors = np.array(anchors).reshape(-1, 2)

    yolo_model = load_model(model_path)

    # Verify model, anchors, and classes are compatible
    num_yolo_classes = len(yolo_class_names)
    num_anchors = len(anchors)
    model_output_channels = yolo_model.layers[-1].output_shape[-1]
    assert model_output_channels == num_anchors * (num_yolo_classes + 5), \
        'Mismatch between model and given anchor and class sizes. ' \
        'Specify matching anchors and classes with --anchors_path and ' \
        '--classes_path flags.'
    print('{} model, anchors, and classes loaded.'.format(model_path))
    model_image_size = yolo_model.layers[0].input_shape[1:3]

    # Generate output tensor targets for filtered bounding boxes.
    yolo_outputs = yolo_head(yolo_model.output, anchors, len(yolo_class_names))
    input_image_shape = K.placeholder(shape=(2, ))
    boxes, scores, classes = yolo_eval(
        yolo_outputs,
        input_image_shape,
        max_boxes=10,
        score_threshold=0.01,
        iou_threshold=args.iou_threshold)

    test_path = os.path.expanduser(args.test_path)
    all_imgs, classes_count, class_mapping = get_data(test_path)
    test_imgs = [s for s in all_imgs if s['imageset'] == 'test']

    if 'bg' not in class_mapping:
        class_mapping['bg'] = len(class_mapping)

    original_mapping = class_mapping
    class_mapping = {v: k for k, v in class_mapping.items()}
    print(class_mapping)

    use_store_data = os.path.expanduser(args.use_store_data)
    measure_mAP_filename = os.path.expanduser(args.measure_mAP_filename)

    if use_store_data == '1':
        print(f"use data in {measure_mAP_filename}")
        with open(measure_mAP_filename, 'rb') as f_measure_mAP:
            TP_obj = pickle.load(f_measure_mAP)
            output_ap(TP_obj[0], TP_obj[1])
            draw_measure_curve(TP_obj[0], TP_obj[1], TP_obj[2], TP_obj[3], original_mapping)
            exit(0)

    T = {}
    P = {}
    T_real = []
    P_real = []

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

            all_dets = []
            for i, c in reversed(list(enumerate(out_classes))):
                predicted_class = yolo_class_names[c]

                if not predicted_class in original_mapping.keys():
                    continue
                box = out_boxes[i]
                score = out_scores[i]

                top, left, bottom, right = box
                top = max(0, np.floor(top + 0.5).astype('int32'))
                left = max(0, np.floor(left + 0.5).astype('int32'))
                bottom = min(image_size[0], np.floor(bottom + 0.5).astype('int32'))
                right = min(image_size[1], np.floor(right + 0.5).astype('int32'))

                det = {'x1': left, 'x2': right, 'y1': top, 'y2': bottom, 'class': predicted_class, 'prob': score}
                all_dets.append(det)

            # Yolov2这个算法已经在输出的时候调了大小了，所以不用缩放
            t, p, t_real, p_real = get_map(all_dets, img_data['bboxes'], (1.0, 1.0), original_mapping)
            T_real += t_real
            P_real += p_real

            for key in t.keys():
                if key not in T:
                    T[key] = []
                    P[key] = []
                T[key].extend(t[key])
                P[key].extend(p[key])

            output_ap(T, P)
        except AssertionError as e:
            _, _, tb = sys.exc_info()
            traceback.print_tb(tb)  # Fixed format
            tb_info = traceback.extract_tb(tb)
            filename, line, func, text = tb_info[-1]

            print('An error occurred on line {} in statement {}'.format(line, text))
        except Exception as e:
            print(f"Some exception occur!\n\r {e} \n we don't care")

    TP_obj = [T, P, T_real, P_real]
    with open(measure_mAP_filename, "wb") as f_measure_mAP:
        pickle.dump(TP_obj, f_measure_mAP)
    draw_measure_curve(T, P, T_real, P_real, original_mapping)

    sess.close()

if __name__ == '__main__':
    ocl.setUseOpenCL(True)
    _main(parser.parse_args())
