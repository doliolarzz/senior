import tensorflow as tf
from tensorflow.keras import layers
from config import config

class WeightLayer(layers.Layer):

    def __init__(self, full_h, full_w):

        super(WeightFinder, self).__init__()

        self.full_h = full_h
        self.full_w = full_w
        self.w = self.add_weight(shape=(config['IMG_SIZE'], config['IMG_SIZE']),
                             initializer='random_normal',
                             trainable=True)
        self.b = self.add_weight(shape=(config['IMG_SIZE'], config['IMG_SIZE']),
                                initializer='zeros',
                                trainable=True)

    # -> In (out_len * N, n_h, n_w, 480, 480) -> Out (out_len * N, height, width)
    def call(self, inputs):
        n, n_h, n_w, _, _ = inputs.shape
        vals = tf.zeros([n, self.full_h, self.full_w], tf.float32)
        counts = tf.zeros([n, self.full_h, self.full_w], tf.float32)
        for h in range(n_h):
            for w in range(n_w):
                h_start = h*config['STRIDE']
                h_end = h*config['STRIDE'] + config['IMG_SIZE']
                w_start = w*config['STRIDE']
                w_end = w*config['STRIDE'] + config['IMG_SIZE']
                vals[:, h_start:h_end, w_start:w_end] += \
                    tf.multiply(inputs[:, h_start:h_end, w_start:w_end], self.w) + self.b
                counts[:, h_start:h_end, w_start:w_end] += self.w

        for h in range(n_h):
            h_start = h*config['STRIDE']
            h_end = h*config['STRIDE'] + config['IMG_SIZE']
            vals[:, h_start:h_end, -config['IMG_SIZE']:] += \
                tf.multiply(inputs[:, h_start:h_end, -config['IMG_SIZE']:], self.w) + self.b
            counts[:, h_start:h_end, -config['IMG_SIZE']:] += self.w

        for w in range(n_w):
            w_start = w*config['STRIDE']
            w_end = w*config['STRIDE'] + config['IMG_SIZE']
            vals[:, -config['IMG_SIZE']:, w_start:w_end] += \
                tf.multiply(inputs[:, -config['IMG_SIZE']:, w_start:w_end], self.w) + self.b
            counts[:, -config['IMG_SIZE']:, w_start:w_end] += self.w
        
        vals[:, -config['IMG_SIZE']:, -config['IMG_SIZE']:] += \
            tf.multiply(inputs[:, -config['IMG_SIZE']:, -config['IMG_SIZE']:], self.w) + self.b
        counts[:, -config['IMG_SIZE']:, -config['IMG_SIZE']:] += self.w

        return tf.divide(vals, counts)

class WeightFinder(tf.keras.Model):

    def __init__(self, full_h, full_w):
        super(WeightFinder, self).__init__()
        self.layer = WeightLayer(full_h, full_w)

    def call(self, inputs):
        x = self.layer(inputs)
        return x