import tensorflow as tf

import tfr_helper
from tf_neiss import InputFnBase


class InputFn2DT(InputFnBase):
    """Input Function Generator for 2d triangle problems,  dataset returns a dict..."""

    def __init__(self, flags):
        super(InputFn2DT, self).__init__()
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
        # print("complex phi in generator", self._flags.complex_phi)
        if not self._flags.complex_phi:
            parsed_dataset = raw_dataset.map(tfr_helper.parse_t2d, num_parallel_calls=10)
        else:
            parsed_dataset = raw_dataset.map(tfr_helper.parse_t2d_phi_complex, num_parallel_calls=10)
        # parsed_dataset = parsed_dataset.shuffle(buffer_size=1000)
        parsed_dataset_batched = parsed_dataset.batch(self._flags.train_batch_size)
        self.dataset = parsed_dataset_batched.repeat()

        return self.dataset.prefetch(100)

    def get_input_fn_val(self):

        with open(self._flags.val_list, "r") as tr_fobj:
            train_filepath_list = [x.strip("\n") for x in tr_fobj.readlines()]

        raw_dataset = tf.data.TFRecordDataset(train_filepath_list)
        if not self._flags.complex_phi:
            parsed_dataset = raw_dataset.map(tfr_helper.parse_t2d, num_parallel_calls=10)
        else:
            parsed_dataset = raw_dataset.map(tfr_helper.parse_t2d_phi_complex, num_parallel_calls=10)
        self.dataset = parsed_dataset.batch(self._flags.val_batch_size)

        return self.dataset.prefetch(2)

    def get_input_fn_file(self, filepath, batch_size=1):
        assert type(filepath) is str
        raw_dataset = tf.data.TFRecordDataset(filepath)
        if not self._flags.complex_phi:
            parsed_dataset = raw_dataset.map(tfr_helper.parse_t2d)
        else:
            parsed_dataset = raw_dataset.map(tfr_helper.parse_t2d_phi_complex)
        self.dataset = parsed_dataset.batch(batch_size)
        return self.dataset.prefetch(100)


if __name__ == "__main__":
    import util.flags as flags
    import trainer.trainer_base  # do not remove, needed for flag imports

    print("run input_fn_generator_2dtriangle debugging...")

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
