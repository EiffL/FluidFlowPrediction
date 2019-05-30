from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import constant_op

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

tfb = tfp.bijectors

__all__ = [
    "Orthogonal"
]

class Orthogonal(tfb.Bijector):
  """Orthogonal matrix Bijector using optimization over the Stiefel manifold.
  """

  def __init__(self,
               input_dim,
               output_dim,
               validate_args=False,
               name=None):
    """Creates the MatvecLU bijector.
    Args:
        Default value: `False`.
      name: Python `str` name given to ops managed by this object.
        Default value: `None` (i.e., "MatvecLU").
    Raises:
      ValueError: If both/neither `channels` and `lower_upper`/`permutation` are
        specified.
    """
    with tf.name_scope(name or 'Orthogonal') as name:
      with tf.variable_scope('%s/orthogonal_stiefel'%name, reuse=tf.AUTO_REUSE):
        self._mat = tf.get_variable('A', [input_dim, output_dim], dtype=tf.float32,
                            initializer=tf.orthogonal_initializer())

    super(Orthogonal, self).__init__(
        is_constant_jacobian=True,
        forward_min_event_ndims=1,
        validate_args=validate_args,
        name=name)

  @property
  def matrix(self):
    return self._mat

  def _forward(self, x):
    return tf.tensordot(x, tf.transpose(self._mat), axes=[[-1], [0]])

  def _inverse(self, y):
    return tf.tensordot(y, self._mat, axes=[[-1], [0]])

  def _forward_log_det_jacobian(self, x):
    return constant_op.constant(0., x.dtype.base_dtype)
