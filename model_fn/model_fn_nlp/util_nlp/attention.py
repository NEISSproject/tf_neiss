import tensorflow as tf


class Selfattention(tf.keras.layers.Layer):
    def __init__(self, params,embed_dim,num_units=128,dropout_rate=0.0,**kwargs):
        super(Selfattention, self).__init__(**kwargs)
        self._flags = params['flags']
        self._num_units=num_units
        self._dropout_rate=dropout_rate
        self.graph_params = dict()
        self._tracked_layers = dict()
        # declare graph_params and update from dict --graph_params
        self.graph_params["ff_query"] = num_units
        self.graph_params["ff_key"] = num_units
        self.graph_params["ff_value"] = embed_dim
        # initilize keras layer
        self._tracked_layers["ff_query"] = tf.keras.layers.Dense(self.graph_params["ff_query"],
                                                                   activation=tf.nn.leaky_relu, name="ff_query")
        self._tracked_layers["ff_key"] = tf.keras.layers.Dense(self.graph_params["ff_key"],
                                                                   activation=tf.nn.leaky_relu, name="ff_key")
        self._tracked_layers["ff_value"] = tf.keras.layers.Dense(self.graph_params["ff_value"],
                                                                   activation=tf.nn.leaky_relu, name="ff_value")
        self._tracked_layers["dropout"] = tf.keras.layers.Dropout(rate=dropout_rate)
        self._tracked_layers["softmax"] = tf.keras.layers.Softmax()
        self._tracked_layers["norm_beta"] = tf.Variable(initial_value=tf.zeros(embed_dim),trainable=True,name="norm_beta")
        self._tracked_layers["norm_gamma"] = tf.Variable(initial_value=tf.ones(embed_dim),trainable=True,name="norm_gamma")


    def call(self, inputs,training, **kwargs):
        # Create querys, keys, values from inputs.
        querys = self._tracked_layers["ff_query"](inputs)
        keys = self._tracked_layers["ff_key"](inputs)
        values = self._tracked_layers["ff_value"](inputs)
        # Multiplication scores contains the matching between EACH (time) query and key value
        # scores[0,10,15] means for the first batch sample, the 10-the query vector matches the 15-th key vector this good
        pre_scores = tf.matmul(querys, tf.transpose(keys, [0, 2, 1]))  # (N, T, T)

        # Scale - divided by the square of num_units to stabalize the gradient
        normed_scores = pre_scores / (self._num_units ** 0.5)

        # Input Masking
        ## Sums the abs values of the features for each timestep, and calculates its sign --> 0 if no activation at all.
        ## key_masks[0,15] indicates whether the 15-th timestep of the first batch sample has to be masked or not
        masks = tf.sign(tf.reduce_sum(tf.abs(inputs), axis=-1))  # (N, T)

        # ## NOT necessary we use broadcasting
        # ## key_masks[0,10,15] indicates whether the score between the 15-th query and 10-th key has to be masked
        masks = tf.tile(tf.expand_dims(masks, 1), [1, tf.shape(masks)[1], 1])  # (N, T, T)
        paddings = tf.ones_like(normed_scores) * (-2 ** 32 + 1)
        masked_scores = tf.where(tf.equal(masks, 0), paddings, normed_scores)  # (N, T, T)


        # Activation
        scores = self._tracked_layers["softmax"](masked_scores)  # (N, T, T)

        # Dropouts
        if self._dropout_rate > 0.0:
            scores = self._tracked_layers["dropout"](scores, training=training)

        # Weighted sum
        outputs = tf.matmul(scores, values)  # ( N, T, C)

        # Residual connection
        outputs += inputs

        # Normalize
        outputs = self._normalize(outputs)  # (N, T_q, C)

        return outputs

    def _normalize(self, inputs,
                   epsilon=1e-8):
        '''Applies layer normalization.

        Args:
          inputs: A tensor with 2 or more dimensions, where the first dimension has
            `batch_size`.
          epsilon: A floating number. A very small number for preventing ZeroDivision Error.
            by the same name.

        Returns:
          A tensor with the same shape and data dtype as `inputs`.
        '''
        mean, variance = tf.nn.moments(inputs, [-1], keepdims=True)
        normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
        outputs = self._tracked_layers["norm_gamma"] * normalized + self._tracked_layers["norm_beta"]

        return outputs

class MultiHeadAttention(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads):
    super(MultiHeadAttention, self).__init__()
    self.num_heads = num_heads
    self.d_model = d_model

    assert d_model % self.num_heads == 0

    self.depth = d_model // self.num_heads

    self.wq = tf.keras.layers.Dense(d_model)
    self.wk = tf.keras.layers.Dense(d_model)
    self.wv = tf.keras.layers.Dense(d_model)

    self.dense = tf.keras.layers.Dense(d_model)

  def split_heads(self, x, batch_size):
    """Split the last dimension into (num_heads, depth).
    Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
    """
    x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
    return tf.transpose(x, perm=[0, 2, 1, 3])

  def call(self, inputs):
    q=inputs['q']
    k=inputs['k']
    v=inputs['v']
    mask=inputs['mask']

    batch_size = tf.shape(q)[0]

    q = self.wq(q)  # (batch_size, seq_len, d_model)
    k = self.wk(k)  # (batch_size, seq_len, d_model)
    v = self.wv(v)  # (batch_size, seq_len, d_model)

    q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
    k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
    v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)


    # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
    # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
    scaled_attention, attention_weights = scaled_dot_product_attention(
        q, k, v, mask)


    scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

    concat_attention = tf.reshape(scaled_attention,
                                  (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

    output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

    return output, attention_weights

def scaled_dot_product_attention(q, k, v, mask):
  """Calculate the attention weights.
  q, k, v must have matching leading dimensions.
  k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
  The mask has different shapes depending on its type(padding or look ahead)
  but it must be broadcastable for addition.

  Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    mask: Float tensor with shape broadcastable
          to (..., seq_len_q, seq_len_k). Defaults to None.

  Returns:
    output, attention_weights
  """

  matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)


  # scale matmul_qk
  scalar=tf.math.reciprocal(tf.math.sqrt(tf.cast(tf.shape(k)[-1], tf.float32)))
  scaled_attention_logits = tf.math.scalar_mul(scalar,matmul_qk)

  # add the mask to the scaled tensor.
  if mask is not None:
    scaled_attention_logits += (mask * -1e9)


  # softmax is normalized on the last axis (seq_len_k) so that the scores
  # add up to 1.
  attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

  output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

  return output, attention_weights

