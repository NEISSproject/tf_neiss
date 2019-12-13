import tensorflow as tf

import input_fn.input_fn_2d.data_gen_2dt.data_gen_t2d_util.tfr_helper as tfr_helper
from input_fn.input_fn_generator_base import InputFnBase


class InputFnRegularPolygon2D(InputFnBase):
    """Input Function Generator for regular polygon 2d problems, dataset returns a dict..."""

    def __init__(self, flags):
        super(InputFnRegularPolygon2D, self).__init__()
        self._flags = flags
        self.iterator = None
        self._next_batch = None
        self.dataset = None

    def get_input_fn_train(self):
        # One instance of train dataset to produce infinite many samples
        assert len(self._flags.train_lists) == 1, "exact one train list is needed for this scenario"
        with open(self._flags.train_lists[0], "r") as tr_fobj:
            train_filepath_list = [x.strip("\n") for x in tr_fobj.readlines()]

        raw_dataset = tf.data.TFRecordDataset(train_filepath_list)
        parsed_dataset = raw_dataset.map(tfr_helper.parse_regular_polygon2d)
        # parsed_dataset = parsed_dataset.shuffle(buffer_size=1000)
        parsed_dataset_batched = parsed_dataset.padded_batch(self._flags.train_batch_size,
                                                             ({'fc': [3, None]}, {'radius': [1], 'rotation': [1],
                                                                                  'translation': [2], 'edges': [1]}),
                                                             ({'fc': tf.constant(0, dtype=tf.float32)},
                                                              {'radius': tf.constant(0, dtype=tf.float32),
                                                               'rotation': tf.constant(0, dtype=tf.float32),
                                                               'translation': tf.constant(0, dtype=tf.float32),
                                                               'edges': tf.constant(0, dtype=tf.int32)}))
        self.dataset = parsed_dataset_batched.repeat()

        return self.dataset.prefetch(5)

    def get_input_fn_val(self):
        with open(self._flags.val_list, "r") as tr_fobj:
            train_filepath_list = [x.strip("\n") for x in tr_fobj.readlines()]

        raw_dataset = tf.data.TFRecordDataset(train_filepath_list)
        parsed_dataset = raw_dataset.map(tfr_helper.parse_regular_polygon2d)
        self.dataset = parsed_dataset.padded_batch(self._flags.val_batch_size,
                                                   ({'fc': [3, None]}, {'radius': [1], 'rotation': [1],
                                                                        'translation': [2], 'edges': [1]}),
                                                   ({'fc': tf.constant(0, dtype=tf.float32)},
                                                    {'radius': tf.constant(0, dtype=tf.float32),
                                                     'rotation': tf.constant(0, dtype=tf.float32),
                                                     'translation': tf.constant(0, dtype=tf.float32),
                                                     'edges': tf.constant(0, dtype=tf.int32)}))

        return self.dataset

    def get_input_fn_file(self, filepath, batch_size=1):
        """input of just one file for use in predict mode to guarantee order"""
        assert type(filepath) is str
        raw_dataset = tf.data.TFRecordDataset(filepath)
        parsed_dataset = raw_dataset.map(tfr_helper.parse_regular_polygon2d)
        self.dataset = parsed_dataset.padded_batch(batch_size,
                                                   ({'fc': [3, None]}, {'radius': [1], 'rotation': [1],
                                                                        'translation': [2], 'edges': [1]}),
                                                   ({'fc': tf.constant(0, dtype=tf.float32)},
                                                    {'radius': tf.constant(0, dtype=tf.float32),
                                                     'rotation': tf.constant(0, dtype=tf.float32),
                                                     'translation': tf.constant(0, dtype=tf.float32),
                                                     'edges': tf.constant(0, dtype=tf.int32)}))
        return self.dataset


if __name__ == "__main__":
    import util.flags as flags

    import trainer.trainer_base  # do not remove, needed for flag imports

    print("run input_fn_generator_regular_polygon debugging...")

    # gen = Generator2dt(flags.FLAGS)
    # for i in range(10):
    #     in_data, tgt = gen.get_data().__next__()
    #     print("output", type(tgt["points"]), in_data["fc"].shape)
    #     # print(tgt["points"])
    #
    # print(os.getcwd())
    # flags.print_flags()
    #
    # input_fn = InputFn2DT(flags.FLAGS)
    # train_dataset = input_fn.get_input_fn_train()
    # counter = 0
    # for i in train_dataset:
    #     counter += 1
    #     if counter >= 10:
    #         break
    #     in_data, tgt = i
    #     print("output", type(tgt["points"]), in_data["fc"].shape)
    #     # print(tgt["points"])
    #
    # print("Done.")
