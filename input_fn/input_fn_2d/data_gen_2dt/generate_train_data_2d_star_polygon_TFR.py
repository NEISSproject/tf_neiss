import datetime
import multiprocessing
import os
import sys
import time
import uuid

import numpy as np

import tensorflow as tf

import util.flags as flags
import input_fn.input_fn_2d.data_gen_2dt.data_gen_t2d_util.tfr_helper as tfr_helper
from util.misc import get_commit_id

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # or any {'0', '1', '2'}
os.environ["CUDA_VISIBLE_DEVICES"] = ""


# ========
flags.define_string("data_id", "magic_synthetic_dataset", "select a name unique name for the dataset")
flags.define_boolean("to_log_file", False,
                     "if set redirect stdout & stderr to this file in data/syntetic_data/<data_id>/<log-file.log>")
flags.define_string("mode", "val", "select 'val' or 'train'")
flags.define_list('files_train_val', int, "[int(train_files), int(val_files)]",
                  'files to generate for train data/val data', default_value=[1000, 10])
flags.define_integer("samples_per_file", 1000, "set number of samples saved in each file")
flags.define_float("d_phi", 0.001, "step between phi's, impacts data size!")
flags.define_integer("jobs", -1, "set number of samples saved in each file")

if __name__ == "__main__":
    main_data_out = "data/synthetic_data/{}".format(flags.FLAGS.data_id)
    original_out = sys.stdout
    original_err = sys.stderr
    if flags.FLAGS.to_log_file:
        logfile_path = os.path.join(main_data_out, "log_{}_{}.txt".format(flags.FLAGS.data_id, flags.FLAGS.mode))
        if not os.path.isdir(os.path.dirname(logfile_path)):
            os.makedirs(os.path.dirname(logfile_path))
        print("redirect messages to: {}".format(logfile_path))
        log_file_object = open(logfile_path, 'w')
        sys.stdout = log_file_object
        sys.stderr = log_file_object

    print("run generat train data 2d regular polygon")
    commit_id, repos_path = get_commit_id(os.path.realpath(__file__))
    print("{} commit-id: {}".format(repos_path, commit_id))
    print("tf-version: {}".format(tf.__version__))
    if flags.FLAGS.mode == "val":
        number_of_files = flags.FLAGS.files_train_val[1]
    else:
        number_of_files = flags.FLAGS.files_train_val[0]
    print("number of files: {}".format(number_of_files))

    flags.print_flags()
    timer1 = time.time()
    dphi = flags.FLAGS.d_phi
    har = 00.0 / 180.0 * np.pi  # hole_half_angle_rad
    mac = 00.0 / 180.0 * np.pi  # max_angle_of_view_cut_rad
    phi_arr = np.concatenate((np.arange(0 + har, np.pi / 2 - mac, dphi),
                              np.arange(np.pi / 2 + har, np.pi - mac, dphi)))
    assert phi_arr.all() == np.arange(0, np.pi, dphi).all()

    samples_per_file = flags.FLAGS.samples_per_file
    data_folder = os.path.join(main_data_out, flags.FLAGS.mode)
    data_points_per_sample = np.ceil(np.pi / dphi).astype(np.int32)
    bytes_to_write = 4 * 3 * data_points_per_sample * number_of_files * flags.FLAGS.samples_per_file / 1024 ** 2
    print("data points per sample: {}".format(data_points_per_sample))
    print("estimated space in MB: {:0.1f}".format(bytes_to_write))
    print("{} samples to generate.".format(number_of_files * flags.FLAGS.samples_per_file))
    if not os.path.isdir(data_folder):
        os.makedirs(data_folder)
    filename_list = list([os.path.join(data_folder, "data_{:07d}.tfr".format(x)) for x in range(number_of_files)])

    print("generating samples...")

    t2d_saver_obj = tfr_helper.StarPolygon2dSaver(epsilon=0.1, phi_arr=phi_arr, samples_per_file=samples_per_file,
                                                  max_size=50, max_edges=6)
    # save_TFR_x = partial(tfr_helper.save_tfr_t2d, samples=flags.FLAGS.samples_per_file)
    # pool = multiprocessing.Pool(1)
    if flags.FLAGS.jobs <= 0:
        jobs = os.cpu_count()
    else:
        jobs = flags.FLAGS.jobs
    pool = multiprocessing.Pool(jobs)
    pool.map(t2d_saver_obj.save_file, filename_list)
    pool.close()
    print("  Time for data generation: {:0.1f}".format(time.time() - timer1))
    print("  Done.")

    print("load&batch-test...")
    timer1 = time.time()
    raw_dataset = tf.data.TFRecordDataset(filename_list)
    print(raw_dataset)
    parsed_dataset = raw_dataset.map(tfr_helper.parse_regular_polygon2d)
    parsed_dataset_batched = parsed_dataset.padded_batch(10, ({'fc': [3, None]},
                                                              {'radius': [1], 'rotation': [1], 'translation': [2],
                                                               'edges': [1]}),
                                                         ({'fc': tf.constant(0, dtype=tf.float32)},
                                                          {'radius': tf.constant(0, dtype=tf.float32),
                                                           'rotation': tf.constant(0, dtype=tf.float32),
                                                           'translation': tf.constant(0, dtype=tf.float32),
                                                           'edges': tf.constant(0, dtype=tf.int32)}))
    # parsed_dataset_batched = parsed_dataset_batched.repeat(10)
    print(parsed_dataset)
    counter = 0

    # sample = parsed_dataset.get_next()
    for sample in parsed_dataset:
        # tf.decode_raw(sample["fc"], out_type=tf.float32)

        a = sample[0]
        b = sample[1]
        counter += 1

    print("input_dict example:", a)
    print("target_dict example:", b)
    print("{} batches loaded.".format(counter))
    print("  Time for load test: {:0.1f}".format(time.time() - timer1))
    print("  Done.")

    print("write list...")
    out_list_name = "lists/{}_{}.lst".format(flags.FLAGS.data_id, flags.FLAGS.mode)
    with open(out_list_name, "w") as file_object:
        file_object.writelines([str(x) + "\n" for x in filename_list])
    print("date+id: {}".format(datetime.datetime.now().strftime('%Y%m-%d%H-%M%S-') + str(uuid.uuid4())))
    print("  Done.")
    if flags.FLAGS.to_log_file:
        sys.stdout = original_out
        sys.stderr = original_err
    print("Finished.")

    # data_path = filename_list
    # with tf.compat.v1.Session() as sess:
    #     feature = {str(prefix) + '/points': tf.FixedLenFeature([], tf.string),
    #                str(prefix) + '/fc': tf.FixedLenFeature([], tf.string)}
    #     # Create a list of filenames and pass it to a queue
    #     filename_queue = tf.train.string_input_producer(data_path, num_epochs=1)
    #     # Define a reader and read the next record
    #     reader = tf.TFRecordReader()
    #     _, serialized_example = reader.read(filename_queue)
    #     # Decode the record read by the reader
    #     features = tf.parse_single_example(serialized_example, features=feature)
    #     # Convert the image data from string back to the numbers
    #     image = tf.decode_raw(features['train/image'], tf.float32)
    #
    #     # Cast label data into int32
    #     label = tf.cast(features['train/label'], tf.int32)
    #     # Reshape image data into the original shape
    #     image = tf.reshape(image, [224, 224, 3])
    #
    #     # Any preprocessing here ...
    #
    #     # Creates batches by randomly shuffling tensors
    #     images, labels = tf.train.shuffle_batch([image, label], batch_size=10, capacity=30, num_threads=1,
    #                                             min_after_dequeue=10)
