import tensorflow as tf

class BatchNormRelu(tf.keras.layers.Layer):  

    def __init__(self,
                 relu=True,
                 init_zero=False,
                 center=True,
                 scale=True,
                 data_format='channels_last',
                 batch_norm_decay=False,
                 BATCH_NORM_EPSILON=1e-5,
                 **kwargs):
        super(BatchNormRelu, self).__init__(**kwargs)
        self.relu = relu
        if init_zero:
            gamma_initializer = tf.zeros_initializer()
        else:
            gamma_initializer = tf.ones_initializer()
        if data_format == 'channels_first':
            axis = 1
        else:
            axis = -1
        #The core part is a BatchNormalization layer. It is written into a class to make it modifiable
        self.bn = tf.keras.layers.BatchNormalization(
            axis=axis,
            momentum=batch_norm_decay,
            epsilon=BATCH_NORM_EPSILON,
            center=center,
            scale=scale,
            fused=False,
            gamma_initializer=gamma_initializer)

    def call(self, inputs, training):
        inputs = self.bn(inputs, training=training)
        if self.relu:
            inputs = tf.nn.relu(inputs)
        return inputs
    
class FixedPadding(tf.keras.layers.Layer): 

    def __init__(self, kernel_size, data_format='channels_last', **kwargs):
        super(FixedPadding, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.data_format = data_format

    def call(self, inputs, training):
        kernel_size = self.kernel_size
        data_format = self.data_format
        pad_total = kernel_size - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        #The core part is to pad the input via the tf.pad. In the pad, the size is pad[D,0]+input[D]+pad[D,1]
        #Compare with the normal pad, it can decide the kernel size and the channels
        if data_format == 'channels_first':
            padded_inputs = tf.pad(
                inputs, [[0, 0], [0, 0], [pad_beg, pad_end], [pad_beg, pad_end]])
        else:
            padded_inputs = tf.pad(
                inputs, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])

        return padded_inputs

class Conv2dFixedPadding(tf.keras.layers.Layer):  # pylint: disable=missing-docstring

    def __init__(self,
                 filters,
                 kernel_size,
                 strides,
                 data_format='channels_last',
                 **kwargs):
        super(Conv2dFixedPadding, self).__init__(**kwargs)
        if strides > 1:
            self.fixed_padding = FixedPadding(kernel_size, data_format=data_format)
        else:
            self.fixed_padding = None
        self.conv2d = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=('SAME' if strides == 1 else 'VALID'),
            use_bias=False,
            kernel_initializer=tf.keras.initializers.VarianceScaling(),
            data_format=data_format)

    def call(self, inputs, training):
        if self.fixed_padding:
            inputs = self.fixed_padding(inputs, training=training)
        return self.conv2d(inputs, training=training)

class IdentityLayer(tf.keras.layers.Layer):

    def call(self, inputs, training):
        return tf.identity(inputs)
    
class SE_Layer(tf.keras.layers.Layer):  # pylint: disable=invalid-name
    """Squeeze and Excitation layer (https://arxiv.org/abs/1709.01507)."""
    """
    SE layers is an new version of conv2D. It is a new method to investigate the channel relationship of the data
    the "squeeze" is the averaging pool along the channel dimension, and form a 1*1*C matrix. After activation, it is "expaned"
    to the input size H*W*C by multiplication method. In this way it can be used in many kinds of model and get better result.
    """

    def __init__(self, filters, se_ratio, data_format='channels_last', **kwargs):
        super(SE_Layer, self).__init__(**kwargs)
        self.data_format = data_format
        self.se_reduce = tf.keras.layers.Conv2D(
            max(1, int(filters * se_ratio)),
            kernel_size=[1, 1],
            strides=[1, 1],
            kernel_initializer=tf.keras.initializers.VarianceScaling(),
            padding='same',
            data_format=data_format,
            use_bias=True)
        self.se_expand = tf.keras.layers.Conv2D(
            None,  # This is filled later in build().
            kernel_size=[1, 1],
            strides=[1, 1],
            kernel_initializer=tf.keras.initializers.VarianceScaling(),
            padding='same',
            data_format=data_format,
            use_bias=True)

    def build(self, input_shape):
        self.se_expand.filters = input_shape[-1]
        super(SE_Layer, self).build(input_shape)

    def call(self, inputs, training):
        spatial_dims = [2, 3] if self.data_format == 'channels_first' else [1, 2]
        se_tensor = tf.reduce_mean(inputs, spatial_dims, keepdims=True)
        se_tensor = self.se_expand(tf.nn.relu(self.se_reduce(se_tensor)))
        return tf.sigmoid(se_tensor) * inputs
    

class SK_Conv2D(tf.keras.layers.Layer):  # pylint: disable=invalid-name
    """Selective kernel convolutional layer (https://arxiv.org/abs/1903.06586)."""

    def __init__(self,
                 filters,
                 strides,
                 sk_ratio,
                 min_dim=32,
                 data_format='channels_last',
                 **kwargs):
        super(SK_Conv2D, self).__init__(**kwargs)
        self.data_format = data_format
        self.filters = filters
        self.sk_ratio = sk_ratio
        self.min_dim = min_dim

        # Two stream convs (using split and both are 3x3).
        self.conv2d_fixed_padding = Conv2dFixedPadding(
            filters=2 * filters,
            kernel_size=3,
            strides=strides,
            data_format=data_format)
        self.batch_norm_relu = BatchNormRelu(data_format=data_format)

        # Mixing weights for two streams.
        mid_dim = max(int(filters * sk_ratio), min_dim)
        self.conv2d_0 = tf.keras.layers.Conv2D(
            filters=mid_dim,
            kernel_size=1,
            strides=1,
            kernel_initializer=tf.keras.initializers.VarianceScaling(),
            use_bias=False,
            data_format=data_format)
        self.batch_norm_relu_1 = BatchNormRelu(data_format=data_format)
        self.conv2d_1 = tf.keras.layers.Conv2D(
            filters=2 * filters,
            kernel_size=1,
            strides=1,
            kernel_initializer=tf.keras.initializers.VarianceScaling(),
            use_bias=False,
            data_format=data_format)
        
    def call(self, inputs, training):
        channel_axis = 1 if self.data_format == 'channels_first' else 3
        pooling_axes = [2, 3] if self.data_format == 'channels_first' else [1, 2]

        # Two stream convs (using split and both are 3x3).
        inputs = self.conv2d_fixed_padding(inputs, training=training)
        inputs = self.batch_norm_relu(inputs, training=training)
        inputs = tf.stack(tf.split(inputs, num_or_size_splits=2, axis=channel_axis))

        # Mixing weights for two streams.
        global_features = tf.reduce_mean(
            tf.reduce_sum(inputs, axis=0), pooling_axes, keepdims=True)
        global_features = self.conv2d_0(global_features, training=training)
        global_features = self.batch_norm_relu_1(global_features, training=training)
        mixing = self.conv2d_1(global_features, training=training)
        mixing = tf.stack(tf.split(mixing, num_or_size_splits=2, axis=channel_axis))
        mixing = tf.nn.softmax(mixing, axis=0)

        return tf.reduce_sum(inputs * mixing, axis=0)