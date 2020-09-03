import argparse
import json
import logging
import os

import tensorflow as tf
import tensorflow_datasets as tfds

from input_fn.input_fn_nlp.input_fn_nlp_base import InputFnNLPBase
from input_fn.input_fn_nlp.util_input_fn_nlp.StringMapper import get_sm

logger = logging.getLogger(os.path.basename(__file__))


class InputFnNER(InputFnNLPBase):
    def __init__(self, flags):
        self.add_types = [tf.int32 if type == "int" else tf.float32 for type in flags.add_types]
        super(InputFnNER, self).__init__(flags)
        logger.info("start init input fct")
        self._tag_string_mapper = get_sm(self._flags.tags)

        self.get_shapes_types_defaults()

        self._tokenizer_de = tfds.features.text.SubwordTextEncoder.load_from_file(self._flags.tokenizer)
        self._tok_vocab_size = self._tokenizer_de.vocab_size

        logger.info("done init input fct")

    def getTagMapper(self):
        return self._tag_string_mapper

    def get_num_tags(self):
        return self._tag_string_mapper.size()

    def get_shapes_types_defaults(self):

        input_shapes = {'sentence': [None]}

        tgt_shapes = {'tgt': [None], 'targetmask': [None]}

        input_types = {'sentence': tf.int32}

        tgt_types = {'tgt': tf.int32, 'targetmask': tf.int32}

        input_defaults = {'sentence': 0}

        tgt_defaults = {'tgt': 0, 'targetmask': 0}

        self._shapes = input_shapes, tgt_shapes
        self._types = input_types, tgt_types
        self._defaults = input_defaults, tgt_defaults

    def _parse_fn(self, training_data):
        sentence = ''
        tags = []
        for j in range(len(training_data)):
            sentence = sentence + training_data[j][0]
            if j < len(training_data) - 1:
                sentence = sentence + ' '
            tags.append(self._tag_string_mapper.get_channel(training_data[j][1]))
        enc_sentence = self._tokenizer_de.encode(sentence)
        tar_real = []
        last_index = None
        curindex = len(enc_sentence) - 1
        for j in range(len(training_data) - 1, -1, -1):
            if last_index is not None:
                curlist = [last_index]
            else:
                curlist = []
            while len(curlist) == 0 or training_data[j][0] not in self._tokenizer_de.decode(curlist):
                curlist = [enc_sentence[curindex]] + curlist
                curindex -= 1
            if last_index is not None:
                tar_real = (len(curlist) - 1) * [tags[j]] + tar_real
                if self._flags.predict_mode:
                    training_data[j].append(len(curlist) - 1)
            else:
                tar_real = len(curlist) * [tags[j]] + tar_real
                if self._flags.predict_mode:
                    training_data[j].append(len(curlist))

            last_subword = self._tokenizer_de.decode([curlist[0]])
            if len(last_subword) > 2 and ' ' in last_subword[1:-1]:
                last_index = curlist[0]
            else:
                last_index = None
        # Add SOS-Tag and EOS-Tag
        enc_sentence = [self._tok_vocab_size] + enc_sentence + [self._tok_vocab_size + 1]
        targetmask = [0] + len(tar_real) * [1] + [0]
        tar_real = [self.get_num_tags()] + tar_real + [self.get_num_tags() + 1]
        if self._flags.predict_mode:
            inputs = {'sentence': [enc_sentence]}
            tgts = {'tgt': [tar_real], 'targetmask': [targetmask]}
            return inputs, tgts, training_data
        inputs = {'sentence': enc_sentence}
        tgts = {'tgt': tar_real, 'targetmask': targetmask}

        return inputs, tgts

    def generator_fn(self):
        for fname in self._fnames:
            with open(fname) as f:
                training_data = json.load(f)
                # print("Yield Sentence")
                for i in range(len(training_data)):
                    yield self._parse_fn(training_data[i])

    def generator_fn_predict(self, fname):
        with open(fname) as f:
            training_data = json.load(f)
            # print("Yield Sentence")
            for i in range(len(training_data)):
                yield self._parse_fn(training_data[i])

    def print_ner_sentence(self, sentence, tgt):
        assert tf.executing_eagerly()
        token_list = [self._tokenizer_de.decode([i]) for i in sentence if
                      i.numpy() < self._tok_vocab_size]
        tag_string = [self._tag_string_mapper.get_value(i.numpy()) for i in tgt if i < self._tag_string_mapper.size()]
        tag_string = [i if i != "UNK" else "*" for i in tag_string]
        assert len(token_list) == len(tag_string)

        format_helper = [max(len(s), len(t)) for s, t in zip(token_list, tag_string)]
        print("|".join([("{:" + str(f) + "}").format(s) for f, s in zip(format_helper, token_list)]))
        print("|".join([("{:" + str(f) + "}").format(t) for f, t in zip(format_helper, tag_string)]))


def test_bad_token(args):
    logger.info(f"Running test_bad_tokens of {logger.name}")
    flags = args
    input_fn_obj = InputFnNER(flags)
    for idx in range(input_fn_obj._tok_vocab_size):
        string = input_fn_obj._tokenizer_de.decode([idx])
        if len(string) > 2 and ' ' in string[1:-1]:
            print(f"{idx:4}\t{string}")
            for char in string:
                assert not char.isalnum(), \
                    "Tokens with space not at start or end, usually contain only special characters"


def test_generator_fn(args):
    logger.info(f"Running test_generator_fn of {logger.name}")
    flags = args
    input_fn_obj = InputFnNER(flags)
    input_fn_obj._mode = 'train'
    input_fn_obj._fnames = []
    for flist in input_fn_obj._flags.train_lists:
        with open(flist, 'r') as f:
            # print(f.read().splitlines())
            input_fn_obj._fnames.extend(f.read().splitlines())
    for idx, batch in enumerate(input_fn_obj.generator_fn()):
        print(batch)
        if idx >= 10:
            break


def main(args):
    logger.info(f"Running main() of {logger.name}")
    input_fn_obj = InputFnNER(args)
    train_dataset = input_fn_obj.get_input_fn_train()

    for idx, batch in enumerate(train_dataset):
        if idx >= 100:
            break
        input_fn_obj.print_ner_sentence(batch[0]['sentence'][0], batch[1]['tgt'][0])


def parse_args(args=None):
    parser = argparse.ArgumentParser(f"Parser of '{logger.name}'")
    parser.add_argument("--train_batch_size", type=int, default=16, help="set train batch size")
    parser.add_argument("--train_lists", type=str, nargs='+', default="lists/ler_train.lst")
    parser.add_argument("--val_list", type=str, default="lists/ler_val.lst")

    parser.add_argument("--tags", type=str, default="data/tags/ler_fg.txt", help='path to tag vocabulary')
    parser.add_argument("--tokenizer", type=str, default="data/tokenizer/tokenizer_de",
                        help="path to a tokenizer file")
    parser.add_argument("--add_types", type=str, default="", help='types that are add features int or float')
    parser.add_argument("--predict_mode", type=bool, default=False)
    parser.add_argument('--buffer', type=int, default=1,
                        help='number of training samples hold in the cache. (effects shuffling)')

    args_ = parser.parse_args(args)
    return args_


if __name__ == "__main__":
    logging.basicConfig()
    logger.setLevel("INFO")
    logger.info(f"Running {logger.name} as __main__...")
    arguments = parse_args()
    # test_generator_fn(args=arguments)
    # main(args=arguments)
    test_bad_token(args=arguments)
