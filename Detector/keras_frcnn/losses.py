from keras import backend as K
from keras.objectives import categorical_crossentropy
import functools

if K.image_dim_ordering() == 'tf':
    import tensorflow as tf

lambda_rpn_regr = 1.0
lambda_rpn_class = 1.0

lambda_cls_regr = 1.0
lambda_cls_class = 1.0

lambda_cls_center_loss_class = 0.5

epsilon = 1e-4

def rpn_loss_regr(num_anchors):
    def rpn_loss_regr_fixed_num(y_true, y_pred):
        if K.image_dim_ordering() == 'th':
            x = y_true[:, 4 * num_anchors:, :, :] - y_pred
            x_abs = K.abs(x)
            x_bool = K.less_equal(x_abs, 1.0)
            return lambda_rpn_regr * K.sum(
                y_true[:, :4 * num_anchors, :, :] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))) / K.sum(epsilon + y_true[:, :4 * num_anchors, :, :])
        else:
            # 请注意这里有要注意，请仔细看y_true代表的是什么意思
            # 这里传进来的y_pred是分两个部分的，同cls
            # 公式:  lambda / Ncls * ∑(y*log(hi) + (1 -y)*log(1 - hi))
            x = y_true[:, :, :, 4 * num_anchors:] - y_pred
            x_abs = K.abs(x)
            x_bool = K.cast(K.less_equal(x_abs, 1.0), tf.float32)

            return lambda_rpn_regr * K.sum(
                y_true[:, :, :, :4 * num_anchors] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))) \
                   / K.sum(epsilon + y_true[:, :, :, :4 * num_anchors])

    return rpn_loss_regr_fixed_num


def rpn_loss_cls(num_anchors):
    def rpn_loss_cls_fixed_num(y_true, y_pred):
        if K.image_dim_ordering() == 'tf':
            # 请注意这里有要注意，请仔细看y_true代表的是什么意思
            # 这里传进来的y_pred是分两个部分的，前半部分是y_true[:, :, :, :num_anchors]表示合法的盒子，后半部分才是neg和pos
            # 公式:  lambda / Ncls * ∑(y*log(hi) + (1 -y)*log(1 - hi))
            return lambda_rpn_class * K.sum(y_true[:, :, :, :num_anchors] * K.binary_crossentropy(y_pred[:, :, :, :], y_true[:, :, :, num_anchors:])) / K.sum(epsilon + y_true[:, :, :, :num_anchors])
        else:
            return lambda_rpn_class * K.sum(y_true[:, :num_anchors, :, :] * K.binary_crossentropy(y_pred[:, :, :, :], y_true[:, num_anchors:, :, :])) / K.sum(epsilon + y_true[:, :num_anchors, :, :])

    return rpn_loss_cls_fixed_num


def class_loss_regr(num_classes):
    def class_loss_regr_fixed_num(y_true, y_pred):
        x = y_true[:, :, 4*num_classes:] - y_pred
        x_abs = K.abs(x)
        x_bool = K.cast(K.less_equal(x_abs, 1.0), 'float32')
        return lambda_cls_regr * K.sum(y_true[:, :, :4*num_classes] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))) \
               / K.sum(epsilon + y_true[:, :, :4*num_classes])
    return class_loss_regr_fixed_num


def class_loss_cls(y_true, y_pred):
    return lambda_cls_class * K.mean(categorical_crossentropy(y_true[0, :, :], y_pred[0, :, :]))

ALPHA_regr = 0.1
lambda_cls_regr_triplet_loss = 2

def class_triplet_loss_regr(num_classes, num_rois):
    def class_loss_regr_fixed_num(y_true, y_pred):
        original_anchor = y_pred[:, :num_rois, :]
        positive_anchor = y_pred[:, num_rois:2*num_rois, :]
        negative_anchor = y_pred[:, 2*num_rois:, :]

        pos_loss = K.sum(
            y_true[:, :num_rois, :4 * num_classes]
            * y_true[:, num_rois:2*num_rois, :4 * num_classes]
            * K.square(original_anchor - positive_anchor), 2)

        neg_loss = K.sum(
            y_true[:, :num_rois, :4 * num_classes]
            * y_true[:, 2*num_rois:, :4 * num_classes]
            * K.square(original_anchor - negative_anchor), 2)

        # y_true[:, :, :4*num_classes]前半部分对于pos和neg应该都是等同的
        return lambda_cls_regr_triplet_loss * K.sum(K.maximum(pos_loss - neg_loss + ALPHA_regr, 0)) \
               / K.sum(epsilon + y_true[:, :num_rois, :4*num_classes])
    return class_loss_regr_fixed_num


ALPHA_cls = 0.5
def class_triplet_loss_cls(num_classes, num_rois):
    def class_loss_cls_fixed_num(y_true, y_pred):
        original_anchor = y_pred[:, :num_rois, :]
        positive_anchor = y_pred[:, num_rois:2*num_rois, :]
        negative_anchor = y_pred[:, 2*num_rois:, :]

        pos_loss = K.sum(K.square(original_anchor - positive_anchor), 2)
        neg_loss = K.sum(K.square(original_anchor - negative_anchor), 2)

        return lambda_cls_class * K.mean(K.maximum(pos_loss - neg_loss + ALPHA_cls,0.0))
    return class_loss_cls_fixed_num
