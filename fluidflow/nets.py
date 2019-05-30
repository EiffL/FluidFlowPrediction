import  tensorflow as tf

def dense_encoder(inputs,
                  latent_size=8, activation=tf.nn.leaky_relu,
                  return_gaussian=False,
                  is_training=True, reuse=tf.AUTO_REUSE, scope='encoder'):
    """
    Encoder using a dense neural network
    """
    with tf.variable_scope(scope, [inputs], reuse=reuse) as sc:
        x = tf.layers.flatten(inputs)
        net = tf.layers.dense(x, 200, activation=tf.nn.leaky_relu)
        net = tf.layers.dense(net, 200, activation=tf.nn.leaky_relu)
        mu = tf.layers.dense(net, latent_size, activation=None)
        if return_gaussian:
            log_sigma = tf.layers.dense(net, latent_size, activation=None)
            return (mu, log_sigma)
        else:
            return mu


def dense_decoder(code,
                  output_size=28,
                  output_channels=1,
                  activation=tf.nn.leaky_relu,
                  is_training=True, reuse=tf.AUTO_REUSE, scope='decoder'):
    """
    Decoder using a dense neural network
    """
    nx = output_size*output_size
    with tf.variable_scope(scope, [code], reuse=reuse) as sc:
        net = tf.layers.dense(code, 200, activation=tf.nn.leaky_relu)
        net = tf.layers.dense(net, 200, activation=tf.nn.leaky_relu)
        net = tf.layers.dense(net, nx, activation=None)
        net = tf.reshape(net, (-1, output_size, output_size, output_channels))
        return net
