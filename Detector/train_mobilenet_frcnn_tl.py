from __future__ import division
import random
import pprint
import sys
import time
import numpy as np
from optparse import OptionParser
import pickle
import traceback

from keras import backend as K
from keras.optimizers import Adam
from keras.layers import Input
from keras.models import Model
from keras_frcnn import config, data_generators
from keras_frcnn import losses as losses
import keras_frcnn.roi_helpers as roi_helpers
from keras.utils import generic_utils
from keras_frcnn import mobilenet as nn
from pascal_voc_parser import get_data
from lxml import etree

"""
frcnn mobile net带triplet loss的版本
"""

def save_result_in_html(mean_overlapping_bboxes, class_acc, loss_rpn_cls, loss_rpn_regr ,loss_class_cls ,loss_class_regr, start_time):
    files = etree.parse("./current.xml")
    data = files.getroot()
    p = etree.SubElement(data, "p")
    p.tail = "\n"

    node = etree.SubElement(p, "mean_overlapping_bboxes")
    node.text = str(mean_overlapping_bboxes)
    node.tail = "\n"

    node = etree.SubElement(p, "class_acc")
    node.text = str(class_acc)
    node.tail = "\n"

    node = etree.SubElement(p, "loss_rpn_cls")
    node.text = str(loss_rpn_cls)
    node.tail = "\n"

    node = etree.SubElement(p, "loss_rpn_regr")
    node.text = str(loss_rpn_regr)
    node.tail = "\n"

    node = etree.SubElement(p, "loss_class_cls")
    node.text = str(loss_class_cls)
    node.tail = "\n"

    node = etree.SubElement(p, "loss_class_regr")
    node.text = str(loss_class_regr)
    node.tail = "\n"

    node = etree.SubElement(p, "Elapsed_time")
    node.text = str(time.time() - start_time)
    node.tail = "\n"

    total_loss = loss_rpn_cls + loss_rpn_regr + loss_class_cls + loss_class_regr
    node = etree.SubElement(p, "total_loss")
    node.text = str(total_loss)
    node.tail = "\n"

    doc = etree.ElementTree(data)
    with open("./current.xml", "wb") as f:
        doc.write(f, pretty_print=True)

def get_training_pack(pos_samples, neg_samples, X2, Y1, Y2):
    # step0:  select C.num_rois / 2 item from pos_samples和neg_smaples as original anchor
    selected_pos_samples = np.random.choice(pos_samples, C.num_rois // 2, replace=True).tolist()
    selected_neg_samples = np.random.choice(neg_samples, C.num_rois // 2, replace=True).tolist()

    triplet_original_samples = selected_pos_samples + selected_neg_samples

    # step1: select positive anchor
    triplet_pos_anchor_samples = []
    for i, sel in enumerate(selected_pos_samples):
        pos = np.where(Y1[0, sel, :] == 1)[0].item()
        pos_samples_ex = pos_samples[np.where(Y1[0, pos_samples, pos] == 1)[0]]
        if len(pos_samples_ex) > 1:
            pos_samples_ex = pos_samples_ex[np.where(pos_samples_ex != sel)]

        assert len(pos_samples_ex) > 0
        val = np.random.choice(pos_samples_ex, 1, replace=False)[0].item()
        triplet_pos_anchor_samples.append(val)

    for i, sel in enumerate(selected_neg_samples):
        neg_samples_ex = selected_neg_samples
        if len(neg_samples_ex) > 1:
            neg_samples_ex = neg_samples[np.where(neg_samples != sel)]

        assert len(neg_samples_ex) > 0
        val = np.random.choice(neg_samples_ex, 1, replace=False)[0].item()
        triplet_pos_anchor_samples.append(val)
 
    # step2: select negative anchor
    triplet_neg_anchor_samples = []
    selected_total_samples = selected_pos_samples + selected_neg_samples
    total_samples = np.hstack((pos_samples, neg_samples))
    for i, sel in enumerate(selected_total_samples):
        pos = np.where(Y1[0, sel, :] == 1)[0].item()
        samples = np.delete(total_samples, np.where(Y1[0, :, pos] == 1)[0], axis=0)

        assert len(samples) > 0
        val = np.random.choice(samples, 1, replace=False)[0].item()
        triplet_neg_anchor_samples.append(val)

    triplet_sel = triplet_original_samples + triplet_pos_anchor_samples + triplet_neg_anchor_samples
    X2_pack = X2[:, triplet_sel, :]
    tripletY1_pack = Y1[:, triplet_sel, :]
    tripletY2_pack = Y2[:, triplet_sel, :]

    return X2_pack, tripletY1_pack, tripletY2_pack


if __name__ == "__main__":
    sys.setrecursionlimit(40000)

    parser = OptionParser()

    parser.add_option("-p", "--path", dest="train_path", help="Path to training data.")
    parser.add_option("-n", "--num_rois", type="int", dest="num_rois", help="Number of RoIs to process at once.", default=32)
    parser.add_option("--hf", dest="horizontal_flips", help="Augment with horizontal flips in training. (Default=false).", action="store_true", default=False)
    parser.add_option("--vf", dest="vertical_flips", help="Augment with vertical flips in training. (Default=false).", action="store_true", default=False)
    parser.add_option("--rot", "--rot_90", dest="rot_90", help="Augment with 90 degree rotations in training. (Default=false).",
                      action="store_true", default=False)
    parser.add_option("--num_epochs", type="int", dest="num_epochs", help="Number of epochs.", default=100)
    parser.add_option("--config_filename", dest="config_filename", help=
                    "Location to store all the metadata related to the training (to be used when testing).",
                    default="config.pickle")
    parser.add_option("--output_weight_path", dest="output_weight_path", help="Output path for weights.", default='./model_frcnn.hdf5')
    parser.add_option("--input_weight_path", dest="input_weight_path", help="Input path for weights. If not specified, will try to load default weights provided by keras.")

    (options, args) = parser.parse_args()

    if not options.train_path:   # if filename is not given
        parser.error('Error: path to training data must be specified. Pass --path to command line')

    # pass the settings from the command line, and persist them in the config object
    C = config.Config()

    C.use_horizontal_flips = bool(options.horizontal_flips)
    C.use_vertical_flips = bool(options.vertical_flips)
    C.rot_90 = bool(options.rot_90)

    C.model_path = options.output_weight_path
    C.num_rois = int(options.num_rois)
    C.network = "mobilenet"

    # check if weight path was passed via command line
    if options.input_weight_path:
        C.base_net_weights = options.input_weight_path
    else:
        raise ValueError("Command line option parser must have base_net_weights")

    all_imgs, classes_count, class_mapping = get_data(options.train_path)

    if 'bg' not in classes_count:
        classes_count['bg'] = 0
        class_mapping['bg'] = len(class_mapping)

    C.class_mapping = class_mapping

    inv_map = {v: k for k, v in class_mapping.items()}
    class_to_color = {class_mapping[v]: np.random.randint(0, 255, 3) for v in class_mapping}

    print('Training images per class:')
    pprint.pprint(classes_count)
    print('Num classes (including bg) = {}'.format(len(classes_count)))

    config_output_filename = options.config_filename

    with open(config_output_filename, 'wb') as config_f:
        pickle.dump(C,config_f)
        print('Config has been written to {}, and can be loaded when testing to ensure correct results'.format(config_output_filename))

    random.shuffle(all_imgs)

    num_imgs = len(all_imgs)

    train_imgs = [s for s in all_imgs if s['imageset'] == 'trainval']
    val_imgs = [s for s in all_imgs if s['imageset'] == 'test']

    print('Num train samples {}'.format(len(train_imgs)))
    print('Num val samples {}'.format(len(val_imgs)))

    data_gen_train = data_generators.get_anchor_gt(train_imgs, classes_count, C, nn.get_img_output_length, K.image_dim_ordering(), mode='train')

    if K.image_dim_ordering() == 'th':
        input_shape_img = (3, None, None)
    else:
        input_shape_img = (None, None, 3)

    img_input = Input(shape=input_shape_img)
    roi_input = Input(shape=(None, 4))

    alpha = 1
    shared_layers = nn.nn_base(img_input, alpha=alpha, trainable=True)

    # define the RPN, built on the base layers
    num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
    rpn = nn.rpn(shared_layers, num_anchors)

    classifier = nn.classifier_triplet_loss(shared_layers, roi_input, C.num_rois, nb_classes=len(classes_count), alpha=alpha, trainable=True)

    model_rpn = Model(img_input, rpn[:2])
    model_classifier = Model([img_input, roi_input], classifier)

    # this is a model that holds both the RPN and the classifier, used to load/save weights for the models
    model_all = Model([img_input, roi_input], rpn[:2] + classifier)

    try:
        print('loading weights from {}'.format(C.base_net_weights))
        model_rpn.load_weights(C.base_net_weights, by_name=True)
        model_classifier.load_weights(C.base_net_weights, by_name=True)
    except:
        print('Could not load pretrained model weights. Weights can be found in the keras application folder \
            https://github.com/fchollet/keras/tree/master/keras/applications')

    optimizer = Adam(lr= 1e-5)
    optimizer_classifier = Adam(lr= 1e-4)
    model_rpn.compile(optimizer=optimizer, loss=[losses.rpn_loss_cls(num_anchors), losses.rpn_loss_regr(num_anchors)])
    model_classifier.compile(optimizer=optimizer_classifier,
                             loss=[losses.class_triplet_loss_cls(len(classes_count), C.num_rois),
                                   losses.class_triplet_loss_regr(len(classes_count)-1, C.num_rois)],
                             metrics={'dense_class_{}'.format(len(classes_count)): 'accuracy'})
    model_all.compile(optimizer='sgd', loss='mae')

    epoch_length = 1000
    num_epochs = int(options.num_epochs)
    iter_num = 0

    losses = np.zeros((epoch_length, 5))
    rpn_accuracy_for_epoch = []
    start_time = time.time()

    best_loss = 1.8663825974439379

    class_mapping_inv = {v: k for k, v in class_mapping.items()}
    print('Starting training')

    vis = True

    for epoch_num in range(num_epochs):
        progbar = generic_utils.Progbar(epoch_length)
        print('Epoch {}/{}'.format(epoch_num + 1, num_epochs))

        while True:
            try:
                X, Y, img_data = next(data_gen_train)

                loss_rpn = model_rpn.train_on_batch(X, Y)

                P_rpn = model_rpn.predict_on_batch(X)
                R = roi_helpers.rpn_to_roi(P_rpn[0], P_rpn[1], C, K.image_dim_ordering(), use_regr=True, overlap_thresh=0.80, max_boxes=300)

                # note: calc_iou converts from (x1,y1,x2,y2) to (x,y,w,h) format
                X2, Y1, Y2, IouS = roi_helpers.calc_iou(R, img_data, C, class_mapping)

                if X2 is None:
                    rpn_accuracy_for_epoch.append(0)
                    continue

                # 最后一个位是bg位
                neg_samples = np.where(Y1[0, :, -1] == 1)
                pos_samples = np.where(Y1[0, :, -1] == 0)

                neg_samples = neg_samples[0] if len(neg_samples) > 0 else []
                pos_samples = pos_samples[0] if len(pos_samples) > 0 else []

                if len(pos_samples) == 0 or len(neg_samples) == 0:
                    print("The iteration should be skipped")
                    continue

                rpn_accuracy_for_epoch.append((len(pos_samples)))

                X2_pack, tripletY1_pack, tripletY2_pack = get_training_pack(pos_samples, neg_samples, X2, Y1, Y2)
                loss_class = model_classifier.train_on_batch([X, X2_pack], [tripletY1_pack, tripletY2_pack])

                losses[iter_num, 0] = loss_rpn[1]
                losses[iter_num, 1] = loss_rpn[2]

                losses[iter_num, 2] = loss_class[1]
                losses[iter_num, 3] = loss_class[2]
                losses[iter_num, 4] = loss_class[3]

                iter_num += 1

                progbar.update(iter_num, [('rpn_cls', np.mean(losses[:iter_num, 0])), ('rpn_regr', np.mean(losses[:iter_num, 1])),
                                          ('detector_cls', np.mean(losses[:iter_num, 2])), ('detector_regr', np.mean(losses[:iter_num, 3]))])

                if iter_num == epoch_length:
                    loss_rpn_cls = np.mean(losses[:, 0])
                    loss_rpn_regr = np.mean(losses[:, 1])
                    loss_class_cls = np.mean(losses[:, 2])
                    loss_class_regr = np.mean(losses[:, 3])
                    class_acc = np.mean(losses[:, 4])

                    mean_overlapping_bboxes = float(sum(rpn_accuracy_for_epoch)) / len(rpn_accuracy_for_epoch)
                    rpn_accuracy_for_epoch = []
                    save_result_in_html(mean_overlapping_bboxes, class_acc, loss_rpn_cls, loss_rpn_regr, loss_class_cls, loss_class_regr, start_time)

                    if C.verbose:
                        print('Mean number of bounding boxes from RPN overlapping ground truth boxes: {}'.format(mean_overlapping_bboxes))
                        print('Classifier accuracy for bounding boxes from RPN: {}'.format(class_acc))
                        print('Loss RPN classifier: {}'.format(loss_rpn_cls))
                        print('Loss RPN regression: {}'.format(loss_rpn_regr))
                        print('Loss Detector classifier: {}'.format(loss_class_cls))
                        print('Loss Detector regression: {}'.format(loss_class_regr))
                        print('Elapsed time: {}'.format(time.time() - start_time))

                    curr_loss = loss_rpn_cls + loss_rpn_regr + loss_class_cls + loss_class_regr
                    print(f"loss = {curr_loss}")
                    iter_num = 0
                    start_time = time.time()

                    if curr_loss < best_loss:
                        if C.verbose:
                            print('Total loss decreased from {} to {}, saving weights'.format(best_loss,curr_loss))
                        best_loss = curr_loss

                    model_all.save_weights(f"model_frcnn_mobilenet_{epoch_num}.hdf5")
                    break
            except AssertionError as e:
                _, _, tb = sys.exc_info()
                traceback.print_tb(tb)  # Fixed format
                tb_info = traceback.extract_tb(tb)
                filename, line, func, text = tb_info[-1]

                print('An error occurred on line {} in statement {}'.format(line, text))
            except Exception as e:
                print('Exception: {}'.format(e))
            if iter_num < epoch_length:
                continue
            else:
                break


    print('Training complete, exiting.')

