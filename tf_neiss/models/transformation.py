import math

import tensorflow as tf


def shape_list(x):
    """Return list of dims, statically where possible."""
    x = tf.convert_to_tensor(x)
    # If unknown rank, return dynamic shape
    if x.get_shape().dims is None:
        return tf.shape(x)

    static = x.get_shape().as_list()
    shape = tf.shape(x)

    ret = []
    for i in range(len(static)):
        dim = static[i]
        if dim is None:
            dim = shape[i]
        ret.append(dim)
    return ret


def cast_like(x, y):
    """Cast x to y's dtype, if necessary."""
    x = tf.convert_to_tensor(x)
    y = tf.convert_to_tensor(y)

    if x.dtype.base_dtype == y.dtype.base_dtype:
        return x

    cast_x = tf.cast(x, y.dtype)
    if cast_x.device != x.device:
        tf.compat.v1.logging.warning("Cast for %s may induce copy from '%s' to '%s'", x.name,
                                     x.device, cast_x.device)
    return cast_x


def get_timing_signal_1d(length,
                         channels,
                         min_timescale=1.0,
                         max_timescale=1.0e4,
                         start_index=0):
    """Gets a bunch of sinusoids of different frequencies.
    Each channel of the input Tensor is incremented by a sinusoid of a different
    frequency and phase.
    This allows attention to learn to use absolute and relative positions.
    Timing signals should be added to some precursors of both the query and the
    memory inputs to attention.
    The use of relative position is possible because sin(x+y) and cos(x+y) can be
    expressed in terms of y, sin(x) and cos(x).
    In particular, we use a geometric sequence of timescales starting with
    min_timescale and ending with max_timescale.  The number of different
    timescales is equal to channels / 2. For each timescale, we
    generate the two sinusoidal signals sin(timestep/timescale) and
    cos(timestep/timescale).  All of these sinusoids are concatenated in
    the channels dimension.
    Args:
      length: scalar, length of timing signal sequence.
      channels: scalar, size of timing embeddings to create. The number of
          different timescales is equal to channels / 2.
      min_timescale: a float
      max_timescale: a float
      start_index: index of first position
    Returns:
      a Tensor of timing signals [length, channels]
    """
    position = tf.cast(tf.range(length) + start_index, dtype=tf.float32)
    num_timescales = channels // 2
    log_timescale_increment = (
            math.log(float(max_timescale) / float(min_timescale)) /
            tf.maximum(tf.cast(num_timescales, dtype=tf.float32) - 1, 1))
    inv_timescales = min_timescale * tf.exp(
        tf.cast(tf.range(num_timescales), dtype=tf.float32) * -log_timescale_increment)
    scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
    signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
    signal = tf.pad(signal, [[0, 0], [0, tf.mod(channels, 2)]])
    return signal


def add_timing_signal_padded(seq_len, seq_max, channels, min_timescale, max_timescale):
    signal = get_timing_signal_1d(seq_len, channels, min_timescale, max_timescale)
    signal = tf.pad(signal, [[0, seq_max - seq_len], [0, 0]])
    return signal


def add_timing_signal_1d_batch(x, seq_lengths,
                               min_timescale=1.0,
                               max_timescale=1.0e4):
    """Adds sinusoids of diff frequencies to a Tensor.
    Args:
      x: a Tensor with shape [batch, max_length, channels]
      seq_lengths: a Tensor with shape [batch] containing the element-wise seq_lengths
      min_timescale: a float
      max_timescale: a float
    Returns:
      a Tensor the same shape as x.
    """
    sh_l = shape_list(x)
    max_len = sh_l[1]
    channels = sh_l[2]
    signal = tf.map_fn(
        lambda seq_len: add_timing_signal_padded(seq_len, max_len, channels, min_timescale, max_timescale),
        elems=seq_lengths, dtype=tf.float32)

    return x + signal


def add_timing_signal_learned_batch(x,
                                    max_len=2000,
                                    zero_pad=False,
                                    scale=False,
                                    scope="time_embedding",
                                    reuse=None):
    sh_l = shape_list(x)
    channels = sh_l[2]
    max_len_batch = sh_l[1]
    batch_size = sh_l[0]

    lookup_ids = tf.tile(tf.expand_dims(tf.range(max_len_batch), 0), [batch_size, 1])
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        lookup_table = tf.compat.v1.get_variable('embed_table',
                                                 dtype=tf.float32,
                                                 shape=[max_len, channels],
                                                 initializer=tf.contrib.layers.xavier_initializer())
        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1, channels]),
                                      lookup_table[1:, :]), 0)

        time_encoding = layers.embedding_lookup(lookup_table, lookup_ids)
        if scale:
            time_encoding = time_encoding * (channels ** 0.5)

        encoded = x + time_encoding
    return encoded
