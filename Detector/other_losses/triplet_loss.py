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

    pos_dist = K.sum(K.square(anchor - positive), 1)
    neg_dist = K.sum(K.square(anchor - negative), 1)

    basic_loss = pos_dist - neg_dist + ALPHA
    loss = K.mean(K.maximum(basic_loss, 0.0), 0)

    return loss