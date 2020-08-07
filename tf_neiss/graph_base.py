import tensorflow as tf


class GraphBase(tf.keras.Model):
    def __init__(self, params):
        super(GraphBase, self).__init__()
        self.graph_params = dict()

        self._flags = params['flags']
        self.global_step = tf.Variable(0, trainable=False, dtype=tf.int64, name="global_step")
        self.global_epoch = tf.Variable(0, trainable=False, dtype=tf.int64, name="global_epoch")
        self._graph_out = None
        self._tracked_layers = dict()
        self._built = False
        self.outputs = None
        self.output_names = None

    def call(self, inputs, training=None, mask=None):
        return self.infer(inputs, is_training=training)

    def print_params(self):
        sorted_dict = sorted(self.graph_params.items(), key=lambda kv: kv[0])
        if len(sorted_dict) > 0:
            print("{}_params:".format("graph"))
            for a in sorted_dict:
                print("  {}: {}".format(a[0], a[1]))

    def get_global_step(self):
        return self._global_step

    def set_interface(self, val_dataset):
        build_inputs = next(iter(val_dataset))[0]
        build_out = self.call(build_inputs, training=False)
        self._set_inputs(inputs=build_inputs, training=False)
        self.outputs = list()
        self.output_names = list()
        for key in sorted(list(self._graph_out)):
            self.outputs.append(build_out[key])
            self.output_names.append(key)
