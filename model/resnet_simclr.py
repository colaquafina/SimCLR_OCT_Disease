from model.resnet_block import *



class BlockGroup(tf.keras.layers.Layer):  # pylint: disable=missing-docstring

    def __init__(self,
                 filters,
                 block_fn,
                 blocks,
                 strides,
                 data_format='channels_last',
                 sk_ratio=0.0,
                 se_ratio=0.0,
                 **kwargs):
        self._name = kwargs.get('name')
        self.sk_ratio = sk_ratio
        self.se_ratio = se_ratio
        super(BlockGroup, self).__init__(**kwargs)

        self.layers = []
        self.layers.append(
            block_fn(
                filters,
                strides,
                use_projection=True,
                data_format=data_format,
                sk_ratio=sk_ratio,
                se_ratio=se_ratio))
        for _ in range(1, blocks):
            self.layers.append(
                block_fn(
                    filters,
                    1,
                    data_format=data_format,
                    sk_ratio=sk_ratio,
                    se_ratio=se_ratio))

    def call(self, inputs, training):
        for layer in self.layers:
            inputs = layer(inputs, training=training)
        return tf.identity(inputs, self._name)

class Resnet(tf.keras.Model):
    """Define base resnet layer"""

    def __init__(self,
                 block_fn,
                 layers,
                 width_multiplier,
                 data_format='channels_last',
                 trainable=True,
                 sk_ratio=0.0,
                 se_ratio=0.0,
                 **kwargs):

        super(Resnet, self).__init__(**kwargs)
        self.data_format = data_format
        self.initial_conv_relu_max_pool = []
        if sk_ratio > 0:  # Use ResNet-C (https://arxiv.org/abs/1812.01187) change the input stem
            self.initial_conv_relu_max_pool.append(
                Conv2dFixedPadding(
                    filters=64 * width_multiplier // 2,
                    kernel_size=3,
                    strides=2,
                    data_format=data_format,
                    trainable=trainable))
            self.initial_conv_relu_max_pool.append(
                BatchNormRelu(data_format=data_format, trainable=trainable))
            self.initial_conv_relu_max_pool.append(
                Conv2dFixedPadding(
                    filters=64 * width_multiplier // 2,
                    kernel_size=3,
                    strides=1,
                    data_format=data_format,
                    trainable=trainable))
            self.initial_conv_relu_max_pool.append(
                BatchNormRelu(data_format=data_format, trainable=trainable))
            self.initial_conv_relu_max_pool.append(
                Conv2dFixedPadding(
                    filters=64 * width_multiplier,
                    kernel_size=3,
                    strides=1,
                    data_format=data_format,
                    trainable=trainable))
            
        else:
            self.initial_conv_relu_max_pool.append(    # the input layers before the residual/bottle neck blocks (https://arxiv.org/abs/1512.03385)
                Conv2dFixedPadding(
                    filters=64 * width_multiplier,
                    kernel_size=7,
                    strides=2,
                    data_format=data_format,
                    trainable=trainable))
        self.initial_conv_relu_max_pool.append(
            IdentityLayer(name='initial_conv', trainable=trainable))
        self.initial_conv_relu_max_pool.append(
            tf.keras.layers.MaxPooling2D(pool_size=(3,3),strides=2)
        )
        self.block_groups = []
        # the block groups is combined by residual of bottle neck blocks
        # It depends of the depth of the resnet model
        # You can see the Resnet-34 as an example in paper (https://arxiv.org/abs/1512.03385)

        self.block_groups.append(
            BlockGroup(
                filters=64 * width_multiplier,
                block_fn=block_fn,
                blocks=layers[0],
                strides=1,
                name='block_group1',
                data_format=data_format,
                se_ratio=se_ratio,
                sk_ratio=sk_ratio,
                trainable=trainable))
        
        self.block_groups.append(
            BlockGroup(
                filters=128 * width_multiplier,
                block_fn=block_fn,  #the name of the fn, such as residual block
                blocks=layers[1],  #the number of the blcok_fn
                strides=2,
                name='block_group2',
                data_format=data_format,
                se_ratio=se_ratio,
                sk_ratio=sk_ratio,
                trainable=trainable))

        self.block_groups.append(
            BlockGroup(
                filters=256 * width_multiplier,
                block_fn=block_fn,
                blocks=layers[2],
                strides=2,
                name='block_group3',
                data_format=data_format,
                se_ratio=se_ratio,
                sk_ratio=sk_ratio,
                trainable=trainable))

        self.block_groups.append(
            BlockGroup(
                filters=512 * width_multiplier,
                block_fn=block_fn,
                blocks=layers[3],
                strides=2,
                name='block_group4',
                data_format=data_format,
                se_ratio=se_ratio,
                sk_ratio=sk_ratio,
                trainable=trainable))
        
    def call(self, inputs, training):
        for layer in self.initial_conv_relu_max_pool:
            inputs = layer(inputs, training=training)

        for i, layer in enumerate(self.block_groups):
            inputs = layer(inputs, training=training)

        inputs = tf.identity(inputs, 'final_conv_block')

        inputs = tf.identity(inputs, 'final_avg_pool')
        return inputs
    

def resnet(resnet_depth,
           width_multiplier,
           data_format='channels_last',
           sk_ratio=0.0,
           se_ratio=0.0):
    # The number below is the depth the the resnet model, for example, Resnet-34 (https://arxiv.org/abs/1512.03385)
    model_params = {
        18: {
            'block': ResidualBlock,
            'layers': [2, 2, 2, 2]
        },
        34: {
            'block': ResidualBlock,
            'layers': [3, 4, 6, 3]
        },
        50: {
            'block': BottleneckBlock,
            'layers': [3, 4, 6, 3]
        },
        101: {
            'block': BottleneckBlock,
            'layers': [3, 4, 23, 3]
        },
        152: {
            'block': BottleneckBlock,
            'layers': [3, 8, 36, 3]
        },
        200: {
            'block': BottleneckBlock,
            'layers': [3, 24, 36, 3]
        }
    }

    if resnet_depth not in model_params:
        raise ValueError('Not a valid resnet_depth:', resnet_depth)

    #Here the params is the depth of the resnet model, including the name of the block and the number of layers
    # According to the params, the Resnet class can form a Resnet model, including the input layers,
    # and the residual or bottle-neck blocks (https://arxiv.org/abs/1512.03385)
    params = model_params[resnet_depth]
    return Resnet(
        params['block'],
        params['layers'],
        width_multiplier,
        data_format=data_format,
        sk_ratio=sk_ratio,
        se_ratio=se_ratio)
