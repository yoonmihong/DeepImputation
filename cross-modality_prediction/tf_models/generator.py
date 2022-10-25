import tensorflow as tf

keras = tf.keras
layers = keras.layers


class Instance_Norm(keras.layers.Layer):
    """ Need to verify """

    def __init__(self, depth, name='instance_norm', trainable=True):
        super(Instance_Norm, self).__init__(name=name)
        self.epsilon = 1e-5

    def call(self, inputs):
        if (tf.__version__) < "2.0":
            mean, variance = tf.nn.moments(inputs, axes=[1, 2, 3], keep_dims=True)
        else:
            mean, variance = tf.nn.moments(inputs, axes=[1, 2, 3], keepdims=True)
        inv = tf.math.rsqrt(variance + self.epsilon)
        normalized = (inputs - mean) * inv
        return normalized


class ResidualBlock(keras.Model):
    """ build a redisual block"""

    def __init__(self, filters, kernel_initializer, norm_type='instance', name='residual_block'):
        super(ResidualBlock, self).__init__(name=name)
        self.conv = layers.Conv3D(filters, 3, kernel_initializer=kernel_initializer, padding='same')
        self.norm = Generator.norm(filters, norm_type)

        self.conv2 = layers.Conv3D(filters, 3, kernel_initializer=kernel_initializer, padding='same')
        self.norm2 = Generator.norm(filters, norm_type)

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.norm(x)
        x = tf.nn.relu(x)

        x = self.conv2(x)
        x = self.norm2(x)

        return inputs + x


class Generator(keras.Model):
    """"""

    def __init__(self, name, conv1_filters, kernel_initializer, norm_type='instance'):
        super(Generator, self).__init__(name=name)

        self.c7s1_32_conv = layers.Conv3D(conv1_filters, 3, strides=(1, 1, 1), padding='same',
                                          kernel_initializer=kernel_initializer)
        self.c7s1_32_norm = self.norm(conv1_filters, norm_type)

        self.dk64_conv = layers.Conv3D(2 * conv1_filters, 3, strides=(2, 2, 2), padding='same',
                                       kernel_initializer=kernel_initializer)
        self.dk64_norm = self.norm(2 * conv1_filters, norm_type)

        self.dk128_conv = layers.Conv3D(4 * conv1_filters, 3, strides=(2, 2, 2), padding='same',
                                        kernel_initializer=kernel_initializer)
        self.dk128_norm = self.norm(4 * conv1_filters, norm_type)

        num_res_block = 6

        self.block_list = []

        for i in range(num_res_block):
            self.block_list.append(ResidualBlock(4 * conv1_filters, kernel_initializer, norm_type))

        self.uk64_deconv = layers.Conv3D(2 * conv1_filters, 3, strides=(1, 1, 1), padding='same',
                                         kernel_initializer=kernel_initializer)

        self.uk64_norm = self.norm(2 * conv1_filters, norm_type)

        self.uk32_deconv = layers.Conv3D(conv1_filters, 3, strides=(1, 1, 1), padding='same',
                                         kernel_initializer=kernel_initializer)

        self.uk32_norm = self.norm(conv1_filters, norm_type)

        self.last_layer = layers.Conv3D(1, 3, strides=(1, 1, 1), padding='same',
                                        kernel_initializer=kernel_initializer)

    @staticmethod
    def norm(filters, norm_type='instance'):
        return Instance_Norm(filters)

    def call(self, inputs):
        x = self.c7s1_32_conv(inputs)
        x = self.c7s1_32_norm(x)
        x = tf.nn.relu(x)

        x = self.dk64_conv(x)
        x = self.dk64_norm(x)
        x = tf.nn.relu(x)

        x = self.dk128_conv(x)
        x = self.dk128_norm(x)
        x = tf.nn.relu(x)

        for block in self.block_list:
            x = block(x)

        x = layers.UpSampling3D(size=(2, 2, 2))(x)
        x = self.uk64_deconv(x)
        x = self.uk64_norm(x)
        x = tf.nn.relu(x)

        x = layers.UpSampling3D(size=(2, 2, 2))(x)
        x = self.uk32_deconv(x)
        x = self.uk32_norm(x)
        x = tf.nn.relu(x)

        x = self.last_layer(x)

        return tf.nn.tanh(x)
