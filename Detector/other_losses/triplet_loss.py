import tensorflow as tf
import numpy as np
from keras import backend as K
from keras.models import Model
from keras.layers import Embedding, Flatten, Input, merge
from keras.optimizers import Adam

ALPHA = 0.2  # Triplet Loss Parameter
# Source: https://github.com/davidsandberg/facenet/blob/master/src/facenet.py
def triplet_loss(x):
    anchor, positive, negative = x

    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), 1)
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), 1)

    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), ALPHA)
    loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0), 0)

    return loss