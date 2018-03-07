"""
MobileNet的简介和具体层的含义请看keras的实现
"""

from __future__ import print_function
from __future__ import absolute_import

from keras.layers import Input, Activation, Flatten, Convolution2D, \
    AveragePooling2D, TimeDistributed, Conv2D, BatchNormalization, Dropout, GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape, Dense
from keras.utils import conv_utils
from keras import initializers, regularizers, constraints
from keras.engine import InputSpec
from keras import backend as K

from keras_frcnn.RoiPoolingConv import RoiPoolingConv

def get_img_output_length(width, height):
    def get_output_length(input_length):
        filter_sizes = [1, 1, 1, 1]
        stride = 2
        for filter_size in filter_sizes:
            input_length = (input_length - filter_size + stride) // stride
        return input_length

    return get_output_length(width), get_output_length(height)

class DepthwiseConv2D(Conv2D):
    def __init__(self,
                 kernel_size,strides=(1, 1),padding='valid',depth_multiplier=1, data_format=None,
                 activation=None,use_bias=True,
                 depthwise_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 depthwise_regularizer=None, bias_regularizer=None,activity_regularizer=None,depthwise_constraint=None,bias_constraint=None,
                 trainable = True,
                 **kwargs):
        super(DepthwiseConv2D, self).__init__(
            filters=None,kernel_size=kernel_size,strides=strides,padding=padding,data_format=data_format,
            activation=activation,use_bias=use_bias,bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,bias_constraint=bias_constraint,
            trainable = trainable,
            **kwargs)
        self.depth_multiplier = depth_multiplier
        self.depthwise_initializer = initializers.get(depthwise_initializer)
        self.depthwise_regularizer = regularizers.get(depthwise_regularizer)
        self.depthwise_constraint = constraints.get(depthwise_constraint)
        self.bias_initializer = initializers.get(bias_initializer)
        self.depthwise_kernel = None
        self.bias = None
        self.trainable = True

    def build(self, input_shape):
        if len(input_shape) < 4:
            raise ValueError('Inputs to `DepthwiseConv2D` should have rank 4. '
                             'Received input shape:', str(input_shape))
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = 3
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs to '
                             '`DepthwiseConv2D` '
                             'should be defined. Found `None`.')
        input_dim = int(input_shape[channel_axis])
        depthwise_kernel_shape = (self.kernel_size[0],
                                  self.kernel_size[1],
                                  input_dim,
                                  self.depth_multiplier)

        self.depthwise_kernel = self.add_weight(
            shape=depthwise_kernel_shape,
            initializer=self.depthwise_initializer, # 初始化的值
            name='depthwise_kernel',
            regularizer=self.depthwise_regularizer, # 正则化
            constraint=self.depthwise_constraint    # 约束
        )

        if self.use_bias:
            self.bias = self.add_weight(shape=(input_dim * self.depth_multiplier,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        # Set input spec.
        self.input_spec = InputSpec(ndim=4, axes={channel_axis: input_dim})
        self.built = True

    def call(self, inputs, training=None):
        # depthwise_conv2d后端已经实现了
        outputs = K.depthwise_conv2d(
            inputs,
            self.depthwise_kernel,
            strides=self.strides,
            padding=self.padding,
            dilation_rate=self.dilation_rate,
            data_format=self.data_format)

        if self.bias:
            outputs = K.bias_add(
                outputs,
                self.bias,
                data_format=self.data_format)

        if self.activation is not None:
            return self.activation(outputs)

        return outputs

    def compute_output_shape(self, input_shape):
        rows, cols, out_filters = (-1, -1, -1)
        if self.data_format == 'channels_first':
            rows = input_shape[2]
            cols = input_shape[3]
            out_filters = input_shape[1] * self.depth_multiplier
        elif self.data_format == 'channels_last':
            rows = input_shape[1]
            cols = input_shape[2]
            out_filters = input_shape[3] * self.depth_multiplier

        rows = conv_utils.conv_output_length(rows, self.kernel_size[0],
                                             self.padding,
                                             self.strides[0])
        cols = conv_utils.conv_output_length(cols, self.kernel_size[1],
                                             self.padding,
                                             self.strides[1])

        if self.data_format == 'channels_first':
            return input_shape[0], out_filters, rows, cols
        elif self.data_format == 'channels_last':
            return input_shape[0], rows, cols, out_filters

    def get_config(self):
        config = super(DepthwiseConv2D, self).get_config()
        config.pop('filters')
        config.pop('kernel_initializer')
        config.pop('kernel_regularizer')
        config.pop('kernel_constraint')
        config['depth_multiplier'] = self.depth_multiplier
        config['depthwise_initializer'] = initializers.serialize(self.depthwise_initializer)
        config['depthwise_regularizer'] = regularizers.serialize(self.depthwise_regularizer)
        config['depthwise_constraint'] = constraints.serialize(self.depthwise_constraint)
        return config

def relu6(x):
    return K.relu(x, max_value=6)

def _depthwise_separable_conv_block(inputs, pointwise_conv_filters, alpha,
                                    depth_multiplier=1, strides=(1, 1), block_id=1, trainable = True):
    # 论文提到的那个Depthwise separable convolution
    # -> Depthwise separable convolution are made up of two layers: depthwise convolutions and pointwise convolutions.
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    # MobileNet提速的关键之一：Width Multiplier（也就是下面的alpha）
    pointwise_conv_filters = int(pointwise_conv_filters * alpha)

    # 论文中的Depthwise还有个rho参数，但是这里没有，kearas是对输入图像的第三个通道进行了处理
    x =  DepthwiseConv2D((3, 3),
                        padding='same',
                        depth_multiplier=depth_multiplier,
                        strides=strides,
                        use_bias=False,
                        trainable= trainable,
                        name='conv_dw_%d' % block_id)(inputs)

    # 请看Batch Normalization那篇论文（arXiv:1502.03167）
    x = BatchNormalization(axis=channel_axis, name='conv_dw_%d_bn' % block_id, trainable=trainable)(x)
    x = Activation(relu6, name='conv_dw_%d_relu' % block_id)(x)

    # 论文中的pointwise convolutions，1x1大小的卷积核
    x = Conv2D(pointwise_conv_filters, (1, 1),
               padding='same',
               use_bias=False,
               strides=(1, 1),
               name='conv_pw_%d' % block_id, trainable=True)(x)
    x = BatchNormalization(axis=channel_axis, name='conv_pw_%d_bn' % block_id, trainable=trainable)(x)
    return Activation(relu6, name='conv_pw_%d_relu' % block_id)(x)

def _depthwise_separable_conv_block_td(inputs, pointwise_conv_filters, alpha,
                                        depth_multiplier=1, input_shape=None, strides=(1, 1), block_id=1, trainable = True):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    pointwise_conv_filters = int(pointwise_conv_filters * alpha)

    if input_shape is not None:
        x = TimeDistributed(DepthwiseConv2D((3, 3), padding='same', depth_multiplier=depth_multiplier, strides=strides,
                            use_bias=False), input_shape=input_shape, name='conv_dw_%d' % block_id, trainable= trainable)(inputs)
    else:
        x = TimeDistributed(DepthwiseConv2D((3, 3), padding='same', depth_multiplier=depth_multiplier, strides=strides,
                            use_bias=False), name='conv_dw_%d' % block_id, trainable= trainable)(inputs)

    x = TimeDistributed(BatchNormalization(axis=channel_axis),name='conv_dw_%d_bn' % block_id, trainable=trainable)(x)
    x = TimeDistributed(Activation(relu6), name='conv_dw_%d_relu' % block_id)(x)

    x = TimeDistributed(Conv2D(pointwise_conv_filters, (1, 1),
               padding='same',
               use_bias=False,
               strides=(1, 1)),
               name='conv_pw_%d' % block_id, trainable=True)(x)
    x = TimeDistributed(BatchNormalization(axis=channel_axis), name='conv_pw_%d_bn' % block_id, trainable=trainable)(x)
    return TimeDistributed(Activation(relu6), name='conv_pw_%d_relu' % block_id)(x)


def _conv_block(inputs, filters, alpha, kernel=(3, 3), strides=(1, 1), trainable = True):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    filters = int(filters * alpha)
    x = Conv2D(filters, kernel,
               padding='same',
               use_bias=False,
               strides=strides,
               name='conv1',
               trainable=trainable)(inputs)
    x = BatchNormalization(axis=channel_axis, name='conv1_bn')(x)
    return Activation(relu6, name='conv1_relu')(x)


def nn_base(input_tensor=None,alpha=1.0, depth_multiplier=1, trainable=False):
    # 为了简化代码，keras的源代码的很多东西我已经删掉了，包括一些检查input_shape的东西
    if K.backend() != 'tensorflow':
        raise RuntimeError('Only TensorFlow backend is currently supported, '
                           'as other backends do not support '
                           'depthwise convolution.')

    if alpha not in [0.25, 0.50, 0.75, 1.0]:
        raise ValueError('If imagenet weights are being loaded, '
                         'alpha can be one of'
                         '`0.25`, `0.50`, `0.75` or `1.0` only.')

    img_input = input_tensor
    # 请仔细看论文那个层的结构，有些feturemap是要降一半的，这里是用stride来实现，而不是池化层

    x = _conv_block(img_input, 32, alpha, strides=(2, 2), trainable=trainable)
    x = _depthwise_separable_conv_block(x, 64, alpha, depth_multiplier, block_id=1)
    x = _depthwise_separable_conv_block(x, 128, alpha, depth_multiplier, strides=(2, 2), block_id=2, trainable=trainable)
    x = _depthwise_separable_conv_block(x, 128, alpha, depth_multiplier, block_id=3, trainable=trainable)
    x = _depthwise_separable_conv_block(x, 256, alpha, depth_multiplier, strides=(2, 2), block_id=4, trainable=trainable)
    x = _depthwise_separable_conv_block(x, 256, alpha, depth_multiplier, block_id=5, trainable=trainable)
    x = _depthwise_separable_conv_block(x, 512, alpha, depth_multiplier, strides=(2, 2), block_id=6, trainable=trainable)
    x = _depthwise_separable_conv_block(x, 512, alpha, depth_multiplier, block_id=7, trainable=trainable)
    x = _depthwise_separable_conv_block(x, 512, alpha, depth_multiplier, block_id=8, trainable=trainable)
    x = _depthwise_separable_conv_block(x, 512, alpha, depth_multiplier, block_id=9, trainable=trainable)
    x = _depthwise_separable_conv_block(x, 512, alpha, depth_multiplier, block_id=10, trainable=trainable)
    x = _depthwise_separable_conv_block(x, 512, alpha, depth_multiplier, block_id=11, trainable=trainable)
    return x


def rpn(base_layers,num_anchors):
    x = Convolution2D(512, (3, 3), padding='same', activation='relu', kernel_initializer='normal', name='rpn_conv1')(base_layers)
    x_class = Convolution2D(num_anchors, (1, 1), activation='sigmoid', kernel_initializer='uniform', name='rpn_out_class')(x)
    x_regr = Convolution2D(num_anchors * 4, (1, 1), activation='linear', kernel_initializer='zero', name='rpn_out_regress')(x)

    return [x_class, x_regr, base_layers]


def classifier(base_layers, input_rois, num_rois, nb_classes, alpha=1.0, depth_multiplier=1, trainable=False):
    # compile times on theano tend to be very high, so we use smaller ROI pooling regions to workaround
    pooling_regions = -1
    input_shape = -1
    if K.backend() == 'tensorflow':
        pooling_regions = 14
        input_shape = (num_rois,14,14,1024)
    elif K.backend() == 'theano':
        pooling_regions = 7
        input_shape = (num_rois,1024,7,7)

    out_roi_pool = RoiPoolingConv(pooling_regions, num_rois)([base_layers, input_rois])

    out = _depthwise_separable_conv_block_td(out_roi_pool, 1024, alpha, depth_multiplier, input_shape, strides=(2, 2), block_id=12, trainable=trainable)
    out = _depthwise_separable_conv_block_td(out, 1024, alpha, depth_multiplier, block_id=13, trainable=trainable)
    out = TimeDistributed(AveragePooling2D((7, 7)), name='avg_pool')(out)
    out = TimeDistributed(Flatten(), name='flatten')(out)
    out_class = TimeDistributed(Dense(nb_classes, activation='softmax', kernel_initializer='zero'), name='dense_class_{}'.format(nb_classes))(out)
    # note: no regression target for bg class
    out_regr = TimeDistributed(Dense(4 * (nb_classes-1), activation='linear', kernel_initializer='zero'), name='dense_regress_{}'.format(nb_classes))(out)
    return [out_class, out_regr]