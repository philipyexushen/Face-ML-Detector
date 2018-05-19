import cv2
import numpy as np
import sys
import pickle
from optparse import OptionParser
from keras import backend as K
from keras.layers import Input
from keras.models import Model
from keras_frcnn import roi_helpers
from pascal_voc_parser import get_data
import keras_frcnn.mobilenet as nn

from common_measure_method import get_map
from common_measure_method import draw_measure_curve
from common_measure_method import output_ap

from img_formater import format_img

if __name__ == "__main__":
    sys.setrecursionlimit(40000)

    parser = OptionParser()

    parser.add_option("-p", "--path", dest="test_path", help="Path to test data.")
    parser.add_option("-n", "--num_rois", dest="num_rois",
                    help="Number of ROIs per iteration. Higher means more memory use.", default=32)
    parser.add_option("--config_filename", dest="config_filename",
                      help="Location to read the metadata related to the training (generated when training).",
                    default="config.pickle")
    parser.add_option("--input_weight_path", dest="input_weight_path",
                      help="Input path for weights. If not specified, will try to load default weights provided by keras.")
    parser.add_option("--measure_mAP_filename", dest="measure_mAP_filename",
                      help="Location to store all the metadata",
                      default="measure_mAP_2.pickle")
    parser.add_option("--use_store_data", dest="use_store_data",
                      help="use_store_data",
                      default="0")

    (options, args) = parser.parse_args()

    if not options.test_path:   # if filename is not given
        parser.error('Error: path to test data must be specified. Pass --path to command line')

    config_output_filename = options.config_filename
    measure_mAP_filename = options.measure_mAP_filename
    use_store_data = options.use_store_data

    with open(config_output_filename, 'rb') as f_in:
        C = pickle.load(f_in)

    # turn off any data augmentation at test time
    C.use_horizontal_flips = False
    C.use_vertical_flips = False
    C.rot_90 = False

    img_path = options.test_path

    class_mapping = C.class_mapping

    if 'bg' not in class_mapping:
        class_mapping['bg'] = len(class_mapping)

    original_mapping = class_mapping
    class_mapping = {v: k for k, v in class_mapping.items()}
    print(class_mapping)
    class_to_color = {class_mapping[v]: np.random.randint(0, 255, 3) for v in class_mapping}
    C.num_rois = int(options.num_rois)

    if use_store_data == '1':
        print(f"use data in {measure_mAP_filename}")
        with open(measure_mAP_filename, 'rb') as f_measure_mAP:
            TP_obj = pickle.load(f_measure_mAP)
            output_ap(TP_obj[0], TP_obj[1])
            draw_measure_curve(TP_obj[0], TP_obj[1], TP_obj[2], TP_obj[3], original_mapping)
            exit(0)

    input_shape_img = (None, None, 3)

    alpha = 1
    num_features = int(512 * alpha)
    input_shape_features = (None, None, num_features)

    # check if weight path was passed via command line
    if options.input_weight_path:
        C.base_net_weights = options.input_weight_path
    else:
        # set the path to weights based on backend and model
        C.base_net_weights = C.model_path

    img_input = Input(shape=input_shape_img)
    roi_input = Input(shape=(C.num_rois, 4))
    feature_map_input = Input(shape=input_shape_features)

    shared_layers = nn.nn_base(img_input, alpha, trainable=True)

    # define the RPN, built on the base layers
    num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
    rpn_layers = nn.rpn(shared_layers, num_anchors)
    classifier = nn.classifier(feature_map_input, roi_input, C.num_rois, nb_classes=len(class_mapping), alpha=alpha, trainable=True)

    model_rpn = Model(img_input, rpn_layers)
    model_classifier_only = Model([feature_map_input, roi_input], classifier)

    model_classifier = Model([feature_map_input, roi_input], classifier)

    model_rpn.load_weights(C.base_net_weights, by_name=True)
    model_classifier.load_weights(C.base_net_weights, by_name=True)

    model_rpn.compile(optimizer='sgd', loss='mse')
    model_classifier.compile(optimizer='sgd', loss='mse')

    all_imgs, _, _ = get_data(options.test_path)
    test_imgs = [s for s in all_imgs if s['imageset'] == 'test']

    T = {}
    P = {}
    T_real = []
    P_real = []

    for idx, img_data in enumerate(test_imgs):
        try:
            print('{}/{}'.format(idx, len(test_imgs)))
            filepath = img_data['filepath']
            print(f"file = {filepath}")

            img = cv2.imread(filepath)

            X, fx, fy = format_img(img, C)

            if K.image_dim_ordering() == 'tf':
                X = np.transpose(X, (0, 2, 3, 1))

            # get the feature maps and output from the RPN
            [Y1, Y2, F] = model_rpn.predict(X)

            R = roi_helpers.rpn_to_roi(Y1, Y2, C, K.image_dim_ordering(), overlap_thresh=0.81, max_boxes=350)

            # convert from (x1,y1,x2,y2) to (x,y,w,h)
            R[:, 2] -= R[:, 0]
            R[:, 3] -= R[:, 1]

            # apply the spatial pyramid pooling to the proposed regions
            bboxes = {}
            probs = {}

            for jk in range(R.shape[0] // C.num_rois + 1):
                ROIs = np.expand_dims(R[C.num_rois * jk:C.num_rois * (jk + 1), :], axis=0)
                if ROIs.shape[1] == 0:
                    break

                if jk == R.shape[0] // C.num_rois:
                    # pad R
                    curr_shape = ROIs.shape
                    target_shape = (curr_shape[0], C.num_rois, curr_shape[2])
                    ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
                    ROIs_padded[:, :curr_shape[1], :] = ROIs
                    ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
                    ROIs = ROIs_padded

                [P_cls, P_regr] = model_classifier_only.predict([F, ROIs])

                for ii in range(P_cls.shape[1]):

                    if np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
                        continue

                    cls_name = class_mapping[np.argmax(P_cls[0, ii, :])]

                    if cls_name not in bboxes:
                        bboxes[cls_name] = []
                        probs[cls_name] = []

                    (x, y, w, h) = ROIs[0, ii, :]

                    cls_num = np.argmax(P_cls[0, ii, :])
                    try:
                        (tx, ty, tw, th) = P_regr[0, ii, 4 * cls_num:4 * (cls_num + 1)]
                        tx /= C.classifier_regr_std[0]
                        ty /= C.classifier_regr_std[1]
                        tw /= C.classifier_regr_std[2]
                        th /= C.classifier_regr_std[3]
                        x, y, w, h = roi_helpers.apply_regr(x, y, w, h, tx, ty, tw, th)
                    except:
                        pass
                    bboxes[cls_name].append([16 * x, 16 * y, 16 * (x + w), 16 * (y + h)])
                    probs[cls_name].append(np.max(P_cls[0, ii, :]))

            all_dets = []

            for key in bboxes:
                bbox = np.array(bboxes[key])

                new_boxes, new_probs = roi_helpers.non_max_suppression_fast(bbox, np.array(probs[key]),
                                                                            overlap_thresh=0.5)
                for jk in range(new_boxes.shape[0]):
                    (x1, y1, x2, y2) = new_boxes[jk, :]
                    det = {'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'class': key, 'prob': new_probs[jk]}
                    all_dets.append(det)

            t, p, t_real, p_real = get_map(all_dets, img_data['bboxes'], (fx, fy), original_mapping)
            T_real += t_real
            P_real += p_real

            for key in t.keys():
                if key not in T:
                    T[key] = []
                    P[key] = []
                T[key].extend(t[key])
                P[key].extend(p[key])

            output_ap(T, P)
        except Exception as e:
            print(f"Some exception occur! But we don`t care, {e}")

    TP_obj = [T, P, T_real, P_real]
    with open(measure_mAP_filename, "wb") as f_measure_mAP:
        pickle.dump(TP_obj, f_measure_mAP)
    draw_measure_curve(T, P, T_real, P_real, original_mapping)
