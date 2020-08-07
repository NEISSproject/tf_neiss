import tensorflow as tf

from tf_neiss import GraphBase
from tf_neiss.flags import update_params


class Graph2D(GraphBase):
    def __init__(self, params):
        super(Graph2D, self).__init__(params)
        self._shape_dynamic = None
        self._use_gpu = True
        if params['num_gpus'] == 0:
            self._use_gpu = False

    def print_params(self):
        self.graph_params = update_params(self.graph_params, self._flags.graph_params, "graph")
        print("{}_params:".format("graph"))
        for i, j in enumerate(self.graph_params):
            print("  {}: {}".format(j, self.graph_params[j]))


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

        # initilize keras layer
        self._tracked_layers["flatten_1"] = tf.keras.layers.Flatten()
        # loop over all number in self.graph_params["dense_layers"]
        for layer_index, n_hidden in enumerate(self.graph_params["dense_layers"]):
            name = "ff_{}".format(layer_index + 1)
            self._tracked_layers[name] = tf.keras.layers.Dense(n_hidden, activation=tf.nn.leaky_relu, name=name)

        self._tracked_layers["ff_final"] = tf.keras.layers.Dense(self.graph_params["nhidden_dense_final"],
                                                                 activation=None, name="ff_final")

        if self.graph_params["edge_classifier"]:
            self._tracked_layers["edge_classifier"] = tf.keras.layers.Dense(self.graph_params["nhidden_max_edges"],
                                                                            activation=tf.nn.softmax,
                                                                            name="edge_classifier")

    @tf.function
    def call(self, inputs, training=False, build=None):
        ff_in = self._tracked_layers["flatten_1"](inputs["fc"])
        if training and self.graph_params["input_dropout"] > 0:
            ff_in = tf.nn.dropout(ff_in, rate=self.graph_params["input_dropout"])
        # loop over all number in self.graph_params["dense_layers"]
        for layer_index, n_hidden in enumerate(self.graph_params["dense_layers"]):
            name = "ff_{}".format(layer_index + 1)
            ff_in = self._tracked_layers[name](ff_in)
            if training and self.graph_params["uniform_noise"] > 0:
                ff_in += tf.random.uniform(tf.shape(ff_in), minval=-self.graph_params["uniform_noise"],
                                           maxval=self.graph_params["uniform_noise"])
            if training and self.graph_params["normal_noise"] > 0:
                ff_in += tf.random.normal(tf.shape(ff_in), stddev=self.graph_params["normal_noise"])

        ff_final = self._tracked_layers["ff_final"](ff_in)
        self._graph_out = {"pre_points": ff_final, "fc": inputs["fc"]}
        edge_final = None
        if self.graph_params["edge_classifier"]:
            edge_final = self._tracked_layers["edge_classifier"](ff_final)

            self._graph_out = {"pre_points": ff_final, "e_pred": edge_final, "fc": inputs["fc"]}

        return self._graph_out
