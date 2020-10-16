import argparse
import json
import logging
import os

import tensorflow as tf
import tensorflow_datasets as tfds

from input_fn.input_fn_nlp.input_fn_nlp_base import InputFnNLPBase
from input_fn.input_fn_nlp.util_input_fn_nlp.StringMapper import get_sm
from util import flags

logger = logging.getLogger(os.path.basename(__file__))

flags.define_string('token_mapper', 'v1', 'how to split tags during tokenization, "v1" or "v2"')


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

        tgt_defaults = {'tgt': self._tag_string_mapper.get_oov_id(), 'targetmask': 0}

        self._shapes = input_shapes, tgt_shapes
        self._types = input_types, tgt_types
        self._defaults = input_defaults, tgt_defaults

    def _parse_fn(self, training_data):
        if self._flags.token_mapper == 'v1':
            return self._parse_sentence_v1(training_data)
        elif self._flags.token_mapper == 'v2':
            return self._parse_sentence_v2(training_data)
        else:
            raise AttributeError("Unknown token maper: {}".format(self._input_params["token_mapper"]))

    def _parse_sentence_v1(self, training_data):
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

    def _parse_sentence_v2(self, training_data):
        tags = []
        tags_se = []  # list of (start, end) positions of the tag
        sentence_string = ''
        # run over all [[pseudo-word, tag], ...]
        for j in range(len(training_data)):
            # do not add a space before or after special characters
            if j > 0 and training_data[j][0][0] not in "])}.,;:!?" and training_data[j - 1][0][0] not in "([{":
                sentence_string = sentence_string + " "
            start_len = len(sentence_string)
            sentence_string = sentence_string + training_data[j][0]
            end_len = len(sentence_string) - 1
            # only add the tag to the list if it is not "O" class
            if self._tag_string_mapper.get_channel(training_data[j][1]) != self._tag_string_mapper.get_oov_id():
                tags.append(self._tag_string_mapper.get_channel(training_data[j][1]))
                tags_se.append((start_len, end_len))

        # encode the hole sentence in whole results in less tokens than encoding by word
        enc_sentence = self._tokenizer_de.encode(sentence_string)

        tokens_se = []  # list of (start, end) positions of the tokens (tag analog to tag_se)
        enc_sentence_string = ''  # construct string from token list piece-wise to count start and end
        for j in range(len(enc_sentence)):
            start_len = len(enc_sentence_string)
            enc_sentence_string = enc_sentence_string + self._tokenizer_de.decode([enc_sentence[j]])
            end_len = len(enc_sentence_string) - 1
            tokens_se.append((start_len, end_len))

        # assign other class to all postions
        tar_real = [self._tag_string_mapper.get_oov_id()] * len(enc_sentence)
        # for each position in token-wise tag list check if a target need to be assigned
        for j in range(len(tar_real)):
            for idx, tag in enumerate(tags):
                # if a token includes the start of a tag
                if tokens_se[j][0] <= tags_se[idx][0] <= tokens_se[j][1]:
                    # add tag without replacement
                    tar_real[j] = tag
                # if the token ends within a tag, may assign I-tag instead of B-tag
                elif tags_se[idx][0] <= tokens_se[j][0] <= tags_se[idx][1]:
                    # change b tag to i tag
                    i_tag = self._tag_string_mapper.get_value(tag).replace("B-", "I-")
                    tar_real[j] = self._tag_string_mapper.get_channel(i_tag)
                # if tag is an I-tag and the token before is a single space assign the I-tag to the space token too
                if self._tag_string_mapper.get_value(tar_real[j]).startswith("I-") and \
                        self._tokenizer_de.decode([enc_sentence[j - 1]]) == " ":
                    tar_real[j - 1] = tar_real[j]

        # Add SOS-Tag and EOS-Tag
        enc_sentence = [self._tok_vocab_size] + enc_sentence + [self._tok_vocab_size + 1]
        targetmask = [0] + len(tar_real) * [1] + [0]
        tar_real = [self.get_num_tags()] + tar_real + [self.get_num_tags() + 1]
        if self._flags.predict_mode:
            inputs = {'sentence': enc_sentence}
            tgts = {'tgt': tar_real, 'targetmask': targetmask}
            return inputs, tgts, training_data
        inputs = {'sentence': enc_sentence}
        tgts = {'tgt': tar_real, 'targetmask': targetmask}
        return inputs, tgts

    def _parse_text_part(self,training_data):
        tags = []
        tags_se = []  # list of (start, end) positions of the tag
        sentence_string = ''
        # run over all [[pseudo-word, tag], ...]
        for j in range(len(training_data)):
            # do not add a space before or after special characters
            if j > 0 and training_data[j][0][0] not in "])}.,;:!?" and training_data[j - 1][0][0] not in "([{":
                sentence_string = sentence_string + " "
            start_len = len(sentence_string)
            sentence_string = sentence_string + training_data[j][0]
            end_len = len(sentence_string) - 1
            # only add the tag to the list if it is not "O" class
            if self._tag_string_mapper.get_channel(training_data[j][1]) != self._tag_string_mapper.get_oov_id():
                tags.append(self._tag_string_mapper.get_channel(training_data[j][1]))
                tags_se.append((start_len, end_len))

        # encode the hole sentence in whole results in less tokens than encoding by word
        enc_sentence = self._tokenizer_de.encode(sentence_string)

        tokens_se = []  # list of (start, end) positions of the tokens (tag analog to tag_se)
        enc_sentence_string = ''  # construct string from token list piece-wise to count start and end
        for j in range(len(enc_sentence)):
            start_len = len(enc_sentence_string)
            enc_sentence_string = enc_sentence_string + self._tokenizer_de.decode([enc_sentence[j]])
            end_len = len(enc_sentence_string) - 1
            tokens_se.append((start_len, end_len))

        # assign other class to all postions
        tar_real = [self._tag_string_mapper.get_oov_id()] * len(enc_sentence)
        # for each position in token-wise tag list check if a target need to be assigned
        for j in range(len(tar_real)):
            for idx, tag in enumerate(tags):
                # if a token includes the start of a tag
                if tokens_se[j][0] <= tags_se[idx][0] <= tokens_se[j][1]:
                    # add tag without replacement
                    tar_real[j] = tag
                # if the token ends within a tag, may assign I-tag instead of B-tag
                elif tags_se[idx][0] <= tokens_se[j][0] <= tags_se[idx][1]:
                    # change b tag to i tag
                    i_tag = self._tag_string_mapper.get_value(tag).replace("B-", "I-")
                    tar_real[j] = self._tag_string_mapper.get_channel(i_tag)
                # if tag is an I-tag and the token before is a single space assign the I-tag to the space token too
                if self._tag_string_mapper.get_value(tar_real[j]).startswith("I-") and \
                        self._tokenizer_de.decode([enc_sentence[j - 1]]) == " ":
                    tar_real[j - 1] = tar_real[j]
        return enc_sentence,tar_real

    def _parse_fn_multi_sent(self, index,article):
        enc_sentence, tar_real = self._parse_text_part(article[index])
        lenprevious=0
        lennext=0
        enc_previous = []
        tar_previous = []
        enc_next = []
        tar_next = []
        if index>0:
            previous=[]
            for i in range(index):
                previous.extend(article[i])
            if len(previous)>self._flags.max_token_text_part:
                previous=previous[len(previous)-self._flags.max_token_text_part:]
            enc_previous, tar_previous = self._parse_text_part(previous)
            lenprevious=len(tar_previous)
        if index<len(article)-1:
            next=[]
            for i in range(len(article)-index-1):
                next.extend(article[index+1+i])
            if len(next)>self._flags.max_token_text_part:
                next=next[:self._flags.max_token_text_part]
            enc_next, tar_next = self._parse_text_part(next)
            lennext=len(tar_next)
        if lenprevious+len(tar_real)+lennext>self._flags.max_token_text_part:
            restlength=self._flags.max_token_text_part-len(tar_real)
            if lenprevious<restlength//2:
                shortenlength=restlength-lenprevious
                enc_next=enc_next[:shortenlength]
                tar_next=tar_next[:shortenlength]
            elif lennext<restlength//2:
                shortenlength=restlength-lennext
                if lenprevious>shortenlength:
                    enc_previous=enc_previous[lenprevious-shortenlength:]
                    tar_previous=tar_previous[lenprevious-shortenlength:]
            else:
                enc_next=enc_next[:restlength//2]
                tar_next=tar_next[:restlength//2]
                enc_previous=enc_previous[lenprevious-restlength//2:]
                tar_previous=tar_previous[lenprevious-restlength//2:]

        # Add SOS-Tag and EOS-Tag
        enc_sentence = [self._tok_vocab_size] + enc_previous + [self._tok_vocab_size+1] + enc_sentence + [self._tok_vocab_size+1] + enc_next + [self._tok_vocab_size + 1]
        targetmask = [0] + len(tar_previous) * [0] + [0] + len(tar_real) * [1] + [0] + len(tar_next) * [0] + [0]
        tar_real = [self.get_num_tags()] + tar_previous + [self.get_num_tags()+1] + tar_real + [self.get_num_tags()+1] + tar_next + [self.get_num_tags() + 1]
        inputs = {'sentence': enc_sentence}
        tgts = {'tgt': tar_real, 'targetmask': targetmask}
        return inputs, tgts

    def generator_fn(self):
        for fname in self._fnames:
            with open(fname) as f:
                if fname.endswith(".txt"):
                    training_data = self.load_txt_sentences(fname)
                elif fname.endswith(".json"):
                    training_data = json.load(f)
                else:
                    raise IOError("Invalid file extension in: '{}', only '.txt' and '.json' is supported".format(fname))
                # print("Yield Sentence")
                if self._flags.multi_sent:
                    for i in range(len(training_data)):
                        yield self._parse_fn_multi_sent(i,training_data)

                else:
                    for i in range(len(training_data)):
                        yield self._parse_fn(training_data[i])

    def load_txt_sentences(self, filename):
        with open(filename) as f:
            text = f.read()
            if "\r" in text:
                raise ValueError(
                    "file '" + filename + "' contains non unix line endings: try dos2unix " + filename)
            training_data = text.strip('\n\t ').split("\n\n")
            list = []
            for sentence in training_data:
                if " " not in sentence:
                    continue
                sentence_split = []
                word_entities = sentence.strip("\n").split("\n")
                for word_entity in word_entities:
                    word_entity_split = word_entity.split(" ")
                    assert len(word_entity_split) == 2
                    sentence_split.append(word_entity_split)
                list.append(sentence_split)
        return list

    def generator_fn_predict(self, fname):
        with open(fname) as f:
            training_data = json.load(f)
            # print("Yield Sentence")
            for i in range(len(training_data)):
                yield self._parse_fn(training_data[i])

    def print_ner_sentence(self, sentence, tgt, mask):
        if tf.is_tensor(sentence):
            assert tf.executing_eagerly()
            sentence = sentence.numpy()
            assert len(sentence.shape) == 1, f"Only single samples supported, received shape: {sentence.shape}"
            tgt = tgt.numpy()
            mask = mask.numpy()


        token_list = []
        for i in sentence:
            if i < self._tok_vocab_size:
                token_list.append(self._tokenizer_de.decode([i]))
            elif i == self._tok_vocab_size:
                token_list.append('<SOS>')
            elif i == self._tok_vocab_size + 1:
                token_list.append('<EOS>')
            else:
                raise IndexError("{} > tok_vocab_size + 1 (which is <EOS>), this is not allowed!".format(i))

        tag_string = [self._tag_string_mapper.get_value(i) if i < self._tag_string_mapper.size() else 'OOB' for i in
                      tgt]
        tag_string = [i if i != "UNK" else "*" for i in tag_string]
        # assert len(token_list) == len(tag_string)

        format_helper = [max(len(s), len(t)) for s, t in zip(token_list, tag_string)]

        tokens_with_visible_space = [x.replace(" ", '\u2423') for x in token_list]
        # tokens = "|".join([("{:" + str(f) + "}").format(s, ) for f, s in zip(format_helper, tokens_with_visible_space)])
        # tags = "|".join([("{:" + str(f) + "}").format(t) for f, t in zip(format_helper, tag_string)])
        # mask = "|".join([("{:" + str(f) + "}").format(t) for f, t in zip(format_helper, mask)])
        # return f'{tokens}\n{tags}\n{mask}'

        tokens = "|".join([("{:" + str(f) + "}").format(s, ) for f, s in zip(format_helper, tokens_with_visible_space)])
        tags = "|".join([("{:" + str(f) + "}").format(t) for f, t in zip(format_helper, tag_string)])
        mask = "|".join([("{:" + str(f) + "}").format(t) for f, t in zip(format_helper, mask)])
        return f'{tokens}\n' \
               f'{tags}\n' \
               f'{mask}'


def test_bad_token(args):
    logger.info("Running test_bad_tokens of {}".format(logger.name))
    flags = args
    input_fn_obj = InputFnNER(flags)
    for idx in range(input_fn_obj._tok_vocab_size):
        string = input_fn_obj._tokenizer_de.decode([idx])
        if len(string) > 2 and ' ' in string[1:-1]:
            print("{:4}\t{}".format(idx, string))
            for char in string:
                assert not char.isalnum(), \
                    "Tokens with space not at start or end, usually contain only special characters"


def test_generator_fn(args):
    logger.info("Running test_generator_fn of {}".format(logger.name))
    flags = args
    input_fn_obj = InputFnNER(flags)
    input_fn_obj._mode = 'train'
    input_fn_obj._fnames = []
    for flist in input_fn_obj._flags.train_lists:
        with open(flist, 'r') as f:
            # print(f.read().splitlines())
            input_fn_obj._fnames.extend(f.read().splitlines())
    for idx, batch in enumerate(input_fn_obj.generator_fn()):
        print(input_fn_obj.print_ner_sentence(batch[0]["sentence"], batch[1]["tgt"], batch[1]["targetmask"]))
        if idx >= 100:
            break


def main(args):
    import numpy as np
    logger.info("Running main() of {}".format(logger.name))
    input_fn_obj_v1 = InputFnNER(args)
    params_dict = dict(**vars(args))
    params_dict['token_mapper'] = 'v2'
    name_space = argparse.Namespace(**params_dict)
    input_fn_obj_v2 = InputFnNER(name_space)
    dataset_v1 = input_fn_obj_v1.get_input_fn_val()
    dataset_v2 = input_fn_obj_v2.get_input_fn_val()

    for idx, (batch_v1, batch_v2) in enumerate(zip(dataset_v1, dataset_v2)):
        if idx >= 100:
            break
        print_cond1 = False
        print_cond2 = False
        backward_other = np.where(np.equal(batch_v1[1]['tgt'][0], input_fn_obj_v1._tag_string_mapper.get_oov_id()), 1, 0)
        forward_other = np.where(np.equal(batch_v2[1]['tgt'][0], input_fn_obj_v2._tag_string_mapper.get_oov_id()), 1, 0)
        if not np.array_equal(backward_other, forward_other):
            print_cond1 = True

        for token in batch_v1[0]['sentence'][0]:
            string = " "
            if input_fn_obj_v1._tokenizer_de.vocab_size > token:
                string = input_fn_obj_v1._tokenizer_de.decode([token])
            if len(string) > 2 and ' ' in string[1:-1]:
                # True if a token with space in the middle is in the sentence
                print_cond2 = True
        if print_cond1:# and print_cond2:
            print("v1")
            print(input_fn_obj_v1.print_ner_sentence(batch_v1[0]['sentence'][0], batch_v1[1]['tgt'][0], batch_v1[1]['targetmask'][0]))
            print("v2")
            print(input_fn_obj_v2.print_ner_sentence(batch_v2[0]['sentence'][0], batch_v2[1]['tgt'][0], batch_v2[1]['targetmask'][0]))



def parse_args(args=None):
    parser = argparse.ArgumentParser("Parser of '{}'".format(logger.name))
    parser.add_argument("--train_batch_size", type=int, default=16, help="set train batch size")
    parser.add_argument("--val_batch_size", type=int, default=16, help="set train batch size")
    parser.add_argument("--train_lists", type=str, nargs='+', default="lists/ler_train.lst")
    parser.add_argument("--val_list", type=str, default="lists/ler_val.lst")
    parser.add_argument("--tags", type=str, default="data/tags/ler_fg.txt", help='path to tag vocabulary')
    parser.add_argument("--tokenizer", type=str, default="data/tokenizer/tokenizer_de",
                        help="path to a tokenizer file")
    parser.add_argument("--add_types", type=str, default="", help='types that are add features int or float')
    parser.add_argument("--token_mapper", type=str, default="v1", help='types that are add features int or float')
    parser.add_argument("--predict_mode", type=bool, default=False)
    parser.add_argument('--buffer', type=int, default=1,
                        help='number of training samples hold in the cache. (effects shuffling)')

    args_ = parser.parse_args(args)
    return args_


if __name__ == "__main__":
    logging.basicConfig()
    logger.setLevel("INFO")
    logger.info("Running {} as __main__...".format(logger.name))
    arguments = parse_args()
    # test_generator_fn(args=arguments)
    main(args=arguments)
    # test_bad_token(args=arguments)
