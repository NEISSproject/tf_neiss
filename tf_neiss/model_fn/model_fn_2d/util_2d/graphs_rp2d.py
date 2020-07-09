import tensorflow as tf

from model_fn.model_fn_2d.util_2d.graphs_2d import Graph2D
import model_fn.util_model_fn.keras_compatible_layers as layers
from util.flags import update_params


class GraphConv2MultiFF(Graph2D):
    def __init__(self, params):
        super(GraphConv2MultiFF, self).__init__(params)
        # v0.1
        self.graph_params["mid_layer_activation"] = "leaky_relu"
        self.graph_params["conv_layer_activation"] = "leaky_relu"
        self.graph_params["input_dropout"] = 0.0
        self.graph_params["batch_norm"] = False
        self.graph_params["dense_layers"] = [256, 128]
        self.graph_params = update_params(self.graph_params, self._flags.graph_params, "graph")

    def infer(self, inputs, is_training):
        if self.graph_params["conv_layer_activation"] == "None":
            conv_layer_activation_fn = None
        else:
            conv_layer_activation_fn = getattr(layers, self.graph_params["conv_layer_activation"])
        if self.graph_params["mid_layer_activation"] == "None":
            mid_layer_activation_fn = None
        else:
            mid_layer_activation_fn = getattr(layers, self.graph_params["mid_layer_activation"])

        fc = tf.cast(inputs['fc'], dtype=tf.float32)
        fc = tf.reshape(fc, [-1, 3, self._flags.data_len, 1])
        # Conv1
        with tf.compat.v1.variable_scope("conv1"):
            kernel_dims = [3, 6, 1, 8]
            conv_strides = [1, 1, 3, 1]
            conv1 =  layers.conv2d_bn_lrn_drop(inputs=fc, kernel_shape=kernel_dims, is_training=is_training,
                                                                strides=conv_strides, activation=conv_layer_activation_fn,
                                                                use_bn=False, use_lrn=False, padding='SAME')
            conv1_len = int((self._flags.data_len + conv_strides[2] - 1) / conv_strides[2])
        # Conv2
        with tf.compat.v1.variable_scope("conv2"):
            kernel_dims = [1, 8, 8, 16]
            conv_strides = [1, 1, 6, 1]
            conv2 =  layers.conv2d_bn_lrn_drop(inputs=conv1, kernel_shape=kernel_dims, is_training=is_training,
                                                                strides=conv_strides, activation=conv_layer_activation_fn,
                                                                use_bn=False, use_lrn=False, padding='SAME')
            conv2_len = int((conv1_len + conv_strides[2] - 1) / conv_strides[2])
        ff_in = tf.reshape(conv2, [-1, conv2_len * 16 * 3])
        ff_in = fc
        for index, nhidden in enumerate(self.graph_params["dense_ keraslayers"]):
            ff_in =  layers.ff_layer(inputs=ff_in, outD=nhidden,
                                                      is_training=is_training, activation=mid_layer_activation_fn,
                                                      use_bn=self.graph_params["batch_norm"], name="ff_{}".format(index + 1))

        ff_final =  layers.ff_layer(inputs=ff_in, outD=self._flags.max_edges * 2,
                                                     is_training=is_training, activation=None, name="ff_final")

        radius_final =  layers.ff_layer(inputs=ff_final,
                                                         outD=1,
                                                         is_training=is_training,
                                                         activation=None,
                                                         name="radius_final")

        rotation_final =  layers.ff_layer(inputs=ff_final,
                                                           outD=1,
                                                           is_training=is_training,
                                                           activation=None,
                                                           name="rotation_final")

        translation_final =  layers.ff_layer(inputs=ff_final,
                                                              outD=2,  # 2 dimension problem
                                                              is_training=is_training,
                                                              activation=None,
                                                              name="translation_final")

        edge_final =  layers.ff_layer(inputs=ff_in,
                                                       outD=self._flags.max_edges - 3,  # at least a triangle!
                                                       is_training=is_training,
                                                       activation= layers.softmax,
                                                       name="edge_final")

        return {"radius_pred": radius_final,
                "rotation_pred": rotation_final,
                "translation_pred": translation_final,
                "edges_pred": edge_final}


class GraphMultiFF(Graph2D):
    def __init__(self, params):
        super(GraphMultiFF, self).__init__(params)
        # v0.2
        self.graph_params["mid_layer_activation"] = "leaky_relu"
        self.graph_params["batch_norm"] = False
        self.graph_params["dense_layers"] = [512, 256, 128, 64]
        self.graph_params["dense_dropout"] = []  # [0.0, 0.0] dropout after each dense layer
        self.graph_params["input_dropout"] = 0.01
        self.graph_params["abs_as_input"] = False
        self.graph_params = update_params(self.graph_params, self._flags.graph_params, "graph")

    def infer(self, inputs, is_training):
        if self.graph_params["mid_layer_activation"] == "None":
            mid_layer_activation_fn = None
        else:
            mid_layer_activation_fn = getattr(layers, self.graph_params["mid_layer_activation"])

        if self.graph_params["abs_as_input"]:
            z_values = tf.slice(inputs['fc'], [0, 1, 0], [-1, 2, -1])
            z_squared = tf.square(z_values)
            abs_in = tf.sqrt(tf.reduce_sum(z_squared, axis=1, keep_dims=True))
            ff_in = tf.stack([abs_in, tf.slice(inputs['fc'], [0, 0, 0], [-1, 1, -1])])
            ff_in = tf.reshape(ff_in, (-1, 2 * self._flags.data_len))
        else:
            ff_in = tf.reshape(inputs['fc'], (-1, 3 * self._flags.data_len))

        if is_training and self.graph_params["input_dropout"] > 0:
            ff_in = tf.nn.dropout(ff_in, keep_prob=1.0 - self.graph_params["input_dropout"])

        for index, nhidden in enumerate(self.graph_params["dense_layers"]):
            ff_in = layers.ff_layer(inputs=ff_in, outD=nhidden,
                                    is_training=is_training, activation=mid_layer_activation_fn,
                                    use_bn=self.graph_params["batch_norm"], name="ff_{}".format(index + 1))

            if is_training and self.graph_params["dense_dropout"] and float(
                    self.graph_params["dense_dropout"][index]) > 0.0:
                ff_in = tf.nn.dropout(ff_in, keep_prob=1.0 - self.graph_params["input_dropout"])

        ff_final = ff_in
        radius_final = layers.ff_layer(inputs=ff_final,
                                       outD=1,
                                       is_training=is_training,
                                       activation=None,
                                       name="radius_final")

        rotation_final = layers.ff_layer(inputs=ff_final,
                                         outD=1,
                                         is_training=is_training,
                                         activation=None,
                                         name="rotation_final")

        translation_final = layers.ff_layer(inputs=ff_final,
                                            outD=2,  # 2 dimension problem
                                            is_training=is_training,
                                            activation=None,
                                            name="translation_final")

        edge_final = layers.ff_layer(inputs=ff_in,
                                     outD=self._flags.max_edges - 3,  # at least a triangle!
                                     is_training=is_training,
                                     activation=layers.softmax,
                                     name="edge_final")

        return {"radius_pred": radius_final,
                "rotation_pred": rotation_final,
                "translation_pred": translation_final,
                "edges_pred": edge_final}
