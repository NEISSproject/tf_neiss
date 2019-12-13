import logging
import collections
import tensorflow as tf
import numpy as np

from model_fn.graph_base import GraphBase
import model_fn.util_model_fn.keras_compatible_layers as layers

from util.flags import update_params


class Graph2D(GraphBase):
    def __init__(self, params):
        super(Graph2D, self).__init__(params)
        self._flags = params['flags']
        self._shape_dynamic = None
        self._use_gpu = True
        if params['num_gpus'] == 0:
            self._use_gpu = False

    def print_params(self):
        self.graph_params = update_params(self.graph_params, self._flags.graph_params, "graph")
        print("{}_params:".format("graph"))
        for i, j in enumerate(self.graph_params):
            print("  {}: {}".format(j, self.graph_params[j]))


class KerasGraphFF3(Graph2D):
    def __init__(self, params):
        super(KerasGraphFF3, self).__init__(params)
        # declare graph_params and update from dict --graph_params
        self.graph_params["ff_hidden_1"] = 128
        self.graph_params["ff_hidden_2"] = 128
        self.graph_params["ff_hidden_3"] = 128
        self.graph_params = update_params(self.graph_params, self._flags.graph_params, "graph")
        # initilize keras layer
        self._tracked_layers["add_layer"] = tf.keras.layers.Add()
        self._tracked_layers["flatten_1"] = tf.keras.layers.Flatten()
        self._tracked_layers["ff_layer_1"] = tf.keras.layers.Dense(self.graph_params["ff_hidden_1"],
                                                                   activation=tf.nn.leaky_relu, name="ff_layer_1")
        self._tracked_layers["ff_layer_2"] = tf.keras.layers.Dense(self.graph_params["ff_hidden_2"],
                                                                   activation=tf.nn.leaky_relu, name="ff_layer_2")
        self._tracked_layers["ff_layer_3"] = tf.keras.layers.Dense(self.graph_params["ff_hidden_3"],
                                                                   activation=tf.nn.leaky_relu, name="ff_layer_3")
        self._tracked_layers["last_layer"] = tf.keras.layers.Dense(6, activation=None, name="last_layer")
        self._tracked_layers["flatten_2"] = tf.keras.layers.Flatten()

    @tf.function
    def call(self, inputs, training=None, mask=None):
        # connect keras layers
        flatten_out = self._tracked_layers["flatten_1"](inputs["fc"])
        ff_layer_1_out = self._tracked_layers["ff_layer_1"](flatten_out)
        ff_layer_2_out = self._tracked_layers["ff_layer_2"](ff_layer_1_out)
        ff_layer_3_out = self._tracked_layers["ff_layer_3"](ff_layer_2_out)
        last_layer_out = self._tracked_layers["last_layer"](ff_layer_3_out)
        final_out = self._tracked_layers["flatten_2"](last_layer_out)
        final_out2 = self._tracked_layers["add_layer"]([final_out, final_out])
        self._graph_out = {"pre_points": final_out, "pre_points2": final_out2}
        return self._graph_out


class GraphMultiFF(Graph2D):
    def __init__(self, params):
        super(GraphMultiFF, self).__init__(params)
        # v0.3
        if not self._flags.complex_phi:
            self.fc_size_0 = 3
        else:
            self.fc_size_0 = 4
        self.graph_params["dense_layers"] = [512, 256, 128, 64, 32]
        self.graph_params["input_dropout"] = 0.0
        self.graph_params["ff_dropout"] = 0.0
        self.graph_params["uniform_noise"] = 0.0
        self.graph_params["normal_noise"] = 0.0
        self.graph_params["nhidden_dense_final"] = 6
        self.graph_params["edge_classifier"] = False
        self.graph_params["batch_norm"] = False
        self.graph_params["nhidden_max_edges"] = 6

        self.graph_params = update_params(self.graph_params, self._flags.graph_params, "graph")

    @tf.function
    def call(self, inputs, training=False, build=None):
        ff_in = layers.reshape(tf.cast(inputs['fc'], dtype=tf.float32),
                                                (-1, int(self.fc_size_0 * self._flags.data_len)))

        if training and self.graph_params["uniform_noise"] > 0:
            ff_in += tf.random.uniform(tf.shape(ff_in), minval=-self.graph_params["uniform_noise"],
                                       maxval=self.graph_params["uniform_noise"])
        if training and self.graph_params["normal_noise"] > 0:
            ff_in += tf.random.normal(tf.shape(ff_in), stddev=self.graph_params["normal_noise"])

        if training and self.graph_params["input_dropout"] > 0:
            print(self.graph_params["input_dropout"])
            ff_in = layers.dropout(ff_in, rate=self.graph_params["input_dropout"], keras_model=self)

        for index, nhidden in enumerate(self.graph_params["dense_layers"]):

            ff_in = layers.ff_layer(inputs=ff_in,
                                                     outD=nhidden,
                                                     is_training=training,
                                                     activation=layers.relu,
                                                     use_bn=self.graph_params["batch_norm"],
                                                     name="ff_{}".format(index + 1),
                                                     keras_model=self)

            if training and self.graph_params["input_dropout"] > 0:
                ff_in = layers.dropout(ff_in, rate=self.graph_params["ff_dropout"], keras_model=self)

        ff_final = layers.ff_layer(inputs=ff_in,
                                                    outD=self.graph_params["nhidden_dense_final"],
                                                    is_training=training,
                                                    use_bn=self.graph_params["batch_norm"],
                                                    activation=None,
                                                    name="ff_final",
                                                    keras_model=self)
        edge_final = None
        if self.graph_params["edge_classifier"]:
            edge_final = layers.ff_layer(inputs=ff_in,
                                                          outD=self.graph_params["nhidden_max_edges"] - 3,  # at least a triangle!
                                                          is_training=training,
                                                          activation=layers.softmax,
                                                          name="edge_final")
        # ff_final = tf.compat.v1.Print(ff_final, [tf.shape(ff_final)])
        self._graph_out = {"pre_points": ff_final, "e_pred": edge_final}
        return self._graph_out


# <editor-fold desc="unmaintened graphs">
# class GraphConv2FF6LSTM2(GraphT2D):
#     def __init__(self, params):
#         super(GraphConv2FF6LSTM2, self).__init__(params)
#         # v0.1
#         self.graph_params["nhidden_dense_1"] = 256
#         self.graph_params["nhidden_dense_2"] = 256
#         self.graph_params["nhidden_dense_3"] = 128
#         self.graph_params["nhidden_dense_4"] = 64
#         self.graph_params["nhidden_dense_5"] = 30
#         self.graph_params["nhidden_lstm_1"] = 16
#         self.graph_params["nhidden_lstm_2"] = 16
#
#         self.graph_params = update_params(self.graph_params, self._flags.graph_params, "graph")
#
#     def infer(self, inputs, training):
#         fc = tf.cast(inputs['fc'], dtype=tf.float32)
#         fc = tf.reshape(fc, [-1, 3, self._flags.data_len])
#         self._shape_dynamic = tf.shape(fc)
#
#         with tf.compat.v1.variable_scope("conv1"):
#             kernel_dims1 = [3, 5, 1, 4]
#             conv_strides1 = [1, 1, 3, 1]
#             conv1 = keraslayers.conv2d_bn_lrn_drop('conv', tf.expand_dims(fc, axis=-1), kernel_dims1, strides=conv_strides1,
#                                               activation=layers.leaky_relu,
#                                               training=training,
#                                               use_bn=False,
#                                               use_lrn=False,
#                                               padding='SAME')
#             conv1_len = int((self._flags.data_len + conv_strides1[2] - 1) / conv_strides1[2])
#         with tf.compat.v1.variable_scope("conv2"):
#             kernel_dims1 = [1, 8, 4, 16]
#             conv_strides1 = [1, 1, 6, 1]
#             conv2 = keraslayers.conv2d_bn_lrn_drop('conv', conv1, kernel_dims1, strides=conv_strides1,
#                                               activation=layers.leaky_relu,
#                                               training=training,
#                                               use_bn=False,
#                                               use_lrn=False,
#                                               padding='SAME')
#             conv2_len = int((conv1_len + conv_strides1[2] - 1) / conv_strides1[2])
#         conv_2_squezze = tf.reshape(conv2, [-1, conv2_len * 16 * 3])
#         # conv_2_squezze = tf.Print(conv_2_squezze, [tf.shape(conv_2_squezze)])
#
#         dense_layer_1 = tf.keras.layers.Dense(self.graph_params["nhidden_dense_1"], activation='relu')
#         dense_res_1 = dense_layer_1(conv_2_squezze)
#         dense_layer_2 = tf.keras.layers.Dense(self.graph_params["nhidden_dense_2"], activation='relu')
#         dense_res_2 = dense_layer_2(dense_res_1)
#         dense_layer_3 = tf.keras.layers.Dense(self.graph_params["nhidden_dense_3"], activation='relu')
#         dense_res_3 = dense_layer_3(dense_res_2)
#         dense_layer_4 = tf.keras.layers.Dense(self.graph_params["nhidden_dense_4"], activation='relu')
#         dense_res_4 = dense_layer_4(dense_res_3)
#         dense_layer_5 = tf.keras.layers.Dense(self.graph_params["nhidden_dense_5"], activation='relu')
#         dense_res_5 = dense_layer_5(dense_res_4)
#
#         # LSTM1
#         rnn_in1 = keraslayers.conv_to_brnn('rnnPre1', tf.expand_dims(
#             tf.reshape(dense_res_5, [-1, int(self.graph_params["nhidden_dense_5"] / 10), 10]), axis=-1))
#
#         rnn_out1 = keraslayers.cublstm_fix('BLSTM1', rnn_in1, 3 * np.ones(self._flags.train_batch_size),
#                                       self.graph_params["nhidden_lstm_1"],
#                                       use_gpu=self._use_gpu, training=training)
#         rnn_in2 = keraslayers.cubrnn_to_cubrnn('rnnPre2', rnn_out1)
#         rnn_out2 = keraslayers.cublstm_fix('BLSTM2', rnn_in2, 3 * np.ones(self._flags.train_batch_size),
#                                       self.graph_params["nhidden_lstm_2"],
#                                       use_gpu=self._use_gpu, training=training)
#
#         rnn_out3 = keraslayers.cubrnn_to_conv('rnnPost3', rnn_out2)
#         rnn_out3_mod = tf.transpose(tf.squeeze(rnn_out3, axis=-1), [0, 2, 1])
#         rnn_out3_mod2 = tf.reduce_sum(rnn_out3_mod, axis=1)
#
#         dense_layer_final = tf.keras.layers.Dense(6, activation=None)
#         dense_final = dense_layer_final(rnn_out3_mod2)
#
#         final_reshape = tf.reshape(dense_final, [-1, 3, 2])
#
#         return {"p_pred": final_reshape}
#
# class GraphConv2FF2(GraphT2D):
#     def __init__(self, params):
#         super(GraphConv2FF2, self).__init__(params)
#         # v0.1
#
#         self.graph_params["nhidden_dense_1"] = 512
#         self.graph_params["nhidden_dense_2"] = 256
#         self.graph_params["nhidden_dense_3"] = 128
#         self.graph_params["nhidden_dense_final"] = 6
#
#         self.graph_params = update_params(self.graph_params, self._flags.graph_params, "graph")
#
#     def infer(self, inputs, training):
#         fc = tf.cast(inputs['fc'], dtype=tf.float32)
#
#         fc = tf.reshape(fc, [-1, 3, 3142])
#
#         with tf.compat.v1.variable_scope("conv1"):
#             kernel_dims1 = [3, 5, 1, 4]
#             conv_strides1 = [1, 1, 3, 1]
#             conv1 = keraslayers.conv2d_bn_lrn_drop('conv', tf.expand_dims(fc, axis=-1), kernel_dims1, strides=conv_strides1,
#                                               activation=layers.leaky_relu,
#                                               training=training,
#                                               use_bn=False,
#                                               use_lrn=False,
#                                               padding='VALID')
#         # Conv2
#         with tf.compat.v1.variable_scope("conv2"):
#             kernel_dims1 = [1, 8, 4, 16]
#             conv_strides1 = [1, 1, 6, 1]
#             conv2 = keraslayers.conv2d_bn_lrn_drop('conv', conv1, kernel_dims1, strides=conv_strides1,
#                                               activation=layers.leaky_relu,
#                                               training=training,
#                                               use_bn=False,
#                                               use_lrn=False,
#                                               padding='SAME')
#
#         conv_2_squezze = tf.reshape(conv2, [-1, 2800])
#         dense_layer_1 = tf.keras.layers.Dense(self.graph_params["nhidden_dense_1"], activation='relu')
#         dense_res_1 = dense_layer_1(conv_2_squezze)
#         dense_layer_2 = tf.keras.layers.Dense(self.graph_params["nhidden_dense_2"], activation='relu')
#
#         dense_res_2 = dense_layer_2(dense_res_1)
#         dense_layer_3 = tf.keras.layers.Dense(self.graph_params["nhidden_dense_3"], activation='linear')
#         dense_res_3 = dense_layer_3(dense_res_2)
#         dense_layer_final = tf.keras.layers.Dense(6, activation=None)
#         dense_final = dense_layer_final(dense_res_3)
#         final_reshape = tf.reshape(dense_final, [-1, 3, 2])
#
#         return {"p_pred": final_reshape}


# class GraphConv2FF6(GraphT2D):
#     def __init__(self, params):
#         super(GraphConv2FF6, self).__init__(params)
#         # v0.1
#         self.graph_params["nhidden_dense_1"] = 512
#         self.graph_params["nhidden_dense_2"] = 256
#         self.graph_params["nhidden_dense_3"] = 128
#         self.graph_params["nhidden_dense_4"] = 64
#         self.graph_params["nhidden_dense_5"] = 32
#         self.graph_params["nhidden_dense_final"] = 6
#
#         self.graph_params = update_params(self.graph_params, self._flags.graph_params, "graph")
#
#     def infer(self, inputs, training):
#         fc = tf.cast(inputs['fc'], dtype=tf.float32)
#         fc = tf.reshape(fc, [-1, 3, self._flags.data_len])
#
#         with tf.compat.v1.variable_scope("conv1"):
#             kernel_dims1 = [3, 5, 1, 4]
#             conv_strides1 = [1, 1, 3, 1]
#             conv1 = keraslayers.conv2d_bn_lrn_drop(tf.expand_dims(fc, axis=-1), kernel_dims1, training=training,
#                                               strides=conv_strides1,
#                                               activation=layers.leaky_relu,
#                                               use_bn=False,
#                                               use_lrn=False,
#                                               padding='SAME')
#             conv1_len = int((self._flags.data_len + conv_strides1[2] - 1) / conv_strides1[2])
#         # print(conv1)
#         # Conv2
#         with tf.compat.v1.variable_scope("conv2"):
#             kernel_dims1 = [1, 8, 4, 16]
#             conv_strides1 = [1, 1, 6, 1]
#             conv2 = keraslayers.conv2d_bn_lrn_drop(conv1, kernel_dims1, training=training,
#                                               strides=conv_strides1,
#                                               activation=layers.leaky_relu,
#                                               use_bn=False,
#                                               use_lrn=False,
#                                               padding='SAME')
#             conv2_len = int((conv1_len + conv_strides1[2] - 1) / conv_strides1[2])
#         conv_2_squezze = tf.reshape(conv2, [-1, conv2_len * 16 * 3])
#         # print(conv2)
#         # conv_2_squezze = tf.reshape(conv2, [-1, 2800])
#         dense_layer_1 = tf.keras.layers.Dense(self.graph_params["nhidden_dense_1"], activation='relu')
#         dense_res_1 = dense_layer_1(conv_2_squezze)
#         dense_layer_2 = tf.keras.layers.Dense(self.graph_params["nhidden_dense_2"], activation='relu')
#         dense_res_2 = dense_layer_2(dense_res_1)
#         dense_layer_3 = tf.keras.layers.Dense(self.graph_params["nhidden_dense_3"], activation='relu')
#         dense_res_3 = dense_layer_3(dense_res_2)
#         dense_layer_4 = tf.keras.layers.Dense(self.graph_params["nhidden_dense_4"], activation='relu')
#         dense_res_4 = dense_layer_4(dense_res_3)
#         dense_layer_5 = tf.keras.layers.Dense(self.graph_params["nhidden_dense_5"], activation='relu')
#         dense_res_5 = dense_layer_5(dense_res_4)
#         dense_layer_final = tf.keras.layers.Dense(self.graph_params["nhidden_dense_final"], activation=None)
#         dense_final = dense_layer_final(dense_res_5)
#         final_reshape = tf.reshape(dense_final, [-1, 3, 2])
#
#         return {"p_pred": final_reshape}
# </editor-fold>

class GraphConv2MultiFF(Graph2D):
    def __init__(self, params):
        super(GraphConv2MultiFF, self).__init__(params)
        # v0.3  # raise version if adjusting default values
        self.graph_params["dense_layers"] = [512, 256, 128, 64, 32]
        # self.graph_params["nhidden_dense_final"] = 6
        self.graph_params["mid_layer_activation"] = "leaky_relu"
        self.graph_params["conv_layer_activation"] = "leaky_relu"
        self.graph_params["input_dropout"] = 0.0
        self.graph_params["batch_norm"] = False
        self.graph_params["edge_classifier"] = False
        self.graph_params["nhidden_max_edges"] = 6
        self.graph_params = update_params(self.graph_params, self._flags.graph_params, "graph")

    def infer(self, inputs, is_training):
        if self.graph_params["conv_layer_activation"] == "None":
            conv_layer_activation_fn = None
        else:
            conv_layer_activation_fn = getattr(layers, self.graph_params["conv_layer_activation"])
        if self.graph_params["mid_layer_activation"] == "None":
            conv_layer_activation_fn = None
        else:
            mid_layer_activation_fn = getattr(layers, self.graph_params["mid_layer_activation"])

        fc = tf.cast(inputs['fc'], dtype=tf.float32)
        fc = tf.reshape(fc, [-1, 3, self._flags.data_len, 1])

        if is_training and self.graph_params["input_dropout"] > 0:
            fc = tf.nn.dropout(fc, keep_prob=1.0 - self.graph_params["input_dropout"])
        # Conv1
        with tf.compat.v1.variable_scope("conv1"):
            kernel_dims = [3, 6, 1, 8]
            conv_strides = [1, 1, 3, 1]
            conv1 = layers.conv2d_bn_lrn_drop(inputs=fc, kernel_shape=kernel_dims, is_training=is_training,
                                                               strides=conv_strides, activation=conv_layer_activation_fn,
                                                               use_bn=False, use_lrn=False, padding='SAME')
            conv1_len = int((self._flags.data_len + conv_strides[2] - 1) / conv_strides[2])
        # Conv2
        with tf.compat.v1.variable_scope("conv2"):
            kernel_dims = [1, 8, 8, 16]
            conv_strides = [1, 1, 6, 1]
            conv2 = layers.conv2d_bn_lrn_drop(inputs=conv1, kernel_shape=kernel_dims, is_training=is_training,
                                                               strides=conv_strides, activation=conv_layer_activation_fn,
                                                               use_bn=False, use_lrn=False, padding='SAME')
            conv2_len = int((conv1_len + conv_strides[2] - 1) / conv_strides[2])
        ff_in = tf.reshape(conv2, [-1, conv2_len * 16 * 3])

        for index, nhidden in enumerate(self.graph_params["dense_layers"]):
            ff_in = layers.ff_layer(inputs=ff_in, outD=nhidden,
                                                     is_training=is_training, activation=mid_layer_activation_fn,
                                                     use_bn=self.graph_params["batch_norm"], name="ff_{}".format(index + 1))

        ff_final = layers.ff_layer(inputs=ff_in, outD=self.graph_params["nhidden_max_edges"] * 2,
                                                    is_training=is_training, activation=None, name="ff_final")
        edge_final = None
        if self.graph_params["edge_classifier"]:
            if 'softmax_crossentropy' == self._flags.loss_mode:
                activation = layers.softmax
                outD = self.graph_params["nhidden_max_edges"] - 2  # one or two points have no area
            elif "abs_diff" == self._flags.loss_mode:
                activation = None
                outD = 1
            edge_final = layers.ff_layer(inputs=ff_in,
                                                          outD=outD,
                                                          is_training=is_training,
                                                          activation=activation,
                                                          name="edge_final")

        def num_step(x, bias):
            return tf.math.divide(1, tf.add(tf.constant(1.0), tf.math.exp(-10000000.0 * (x - bias))))

        ff_final_reshaped = tf.reshape(ff_final, shape=(-1, self.graph_params["nhidden_max_edges"], 2))
        num_step_tensor = 1.0 - num_step(
            tf.cast(tf.range(self.graph_params["nhidden_max_edges"]), dtype=tf.float32) + 1.0,
            (tf.minimum(tf.maximum(edge_final, 0), self.graph_params["nhidden_max_edges"] - 2) + 3.5))
        num_step_tensor = tf.expand_dims(num_step_tensor, axis=-1)
        num_step_tensor_broadcast = tf.broadcast_to(num_step_tensor, [tf.shape(ff_final_reshaped)[0], 6, 2])
        ff_final2 = tf.math.multiply(ff_final_reshaped, num_step_tensor_broadcast)

        # edge_final = tf.Print(edge_final, [tf.shape(num_step_tensor_broadcast), num_step_tensor_broadcast], summarize=1000, message="ff_final2")
        # edge_final = tf.Print(edge_final, [edge_final], summarize=1000, message="edges final:")

        return {"p_pred": ff_final2, "e_pred": edge_final}


class GraphConv2MultiFFTriangle(Graph2D):
    def __init__(self, params):
        super(GraphConv2MultiFFTriangle, self).__init__(params)
        # v0.2  # raise version if adjusting default values
        self.graph_params["dense_layers"] = [512, 256, 128, 64, 32]
        self.graph_params["nhidden_dense_final"] = 6
        self.graph_params["mid_layer_activation"] = "leaky_relu"
        self.graph_params["conv_layer_activation"] = "leaky_relu"
        self.graph_params["batch_norm"] = False
        self.graph_params["edge_classifier"] = False
        self.graph_params["nhidden_max_edges"] = 6
        self.graph_params = update_params(self.graph_params, self._flags.graph_params, "graph")

    def infer(self, inputs, is_training):
        conv_layer_activation_fn = getattr(layers, self.graph_params["conv_layer_activation"])
        mid_layer_activation_fn = getattr(layers, self.graph_params["mid_layer_activation"])

        fc = tf.cast(inputs['fc'], dtype=tf.float32)
        fc = tf.reshape(fc, [-1, 3, self._flags.data_len, 1])
        # Conv1
        with tf.compat.v1.variable_scope("conv1"):
            kernel_dims = [3, 5, 1, 4]
            conv_strides = [1, 1, 3, 1]
            conv1 = layers.conv2d_bn_lrn_drop(inputs=fc, kernel_shape=kernel_dims, is_training=is_training,
                                                               strides=conv_strides, activation=conv_layer_activation_fn,
                                                               use_bn=False, use_lrn=False, padding='SAME')
            conv1_len = int((self._flags.data_len + conv_strides[2] - 1) / conv_strides[2])
        # Conv2
        with tf.compat.v1.variable_scope("conv2"):
            kernel_dims = [1, 8, 4, 16]
            conv_strides = [1, 1, 6, 1]
            conv2 = layers.conv2d_bn_lrn_drop(inputs=conv1, kernel_shape=kernel_dims, is_training=is_training,
                                                               strides=conv_strides, activation=conv_layer_activation_fn,
                                                               use_bn=False, use_lrn=False, padding='SAME')
            conv2_len = int((conv1_len + conv_strides[2] - 1) / conv_strides[2])
        ff_in = tf.reshape(conv2, [-1, conv2_len * 16 * 3])

        for index, nhidden in enumerate(self.graph_params["dense_layers"]):
            ff_in = layers.ff_layer(inputs=ff_in, outD=nhidden,
                                                     is_training=is_training, activation=mid_layer_activation_fn,
                                                     use_bn=self.graph_params["batch_norm"], name="ff_{}".format(index + 1))

        ff_final = layers.ff_layer(inputs=ff_in, outD=self.graph_params["nhidden_dense_final"],
                                                    is_training=is_training, activation=None, name="ff_final")

        return {"p_pred": ff_final}
