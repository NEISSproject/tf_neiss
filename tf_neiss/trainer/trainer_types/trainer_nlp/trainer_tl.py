from trainer.trainer_base import TrainerBase
import model_fn.model_fn_nlp.model_fn_tl as models
import util.flags as flags
import os
from input_fn.input_fn_nlp.input_fn_tl import InputFnTL
import tensorflow as tf
from util.misc import get_commit_id
import matplotlib.pyplot as plt

# Model parameter
# ===============
flags.define_string('model_type', 'ModelTL', 'Model Type to use choose from: ModelTriangle')
flags.define_string('tokenizer_inp', "../../../data/tokenizer/tokenizer_pt", 'path to input tokenizer')
flags.define_string('tokenizer_tar', '../../../data/tokenizer/tokenizer_en', 'path to target tokenizer')
flags.define_string('graph', 'KerasGraphFF3', 'class name of graph architecture')
flags.define_list('add_types', str, 'types that are add features int or float',
                  "", [])
flags.define_integer('buffer', 20000,
                     'number of training samples hold in the cache. (effects shuffling)')
flags.define_boolean('predict_mode', False, 'If and only if true the prediction will be accomplished')
flags.FLAGS.parse_flags()


class TrainerTL(TrainerBase):
    def __init__(self):
        super(TrainerTL, self).__init__()
        self._input_fn_generator = InputFnTL(self._flags)
        self._input_fn_generator.print_params()

        self._params['input_vocab_size'] = self._input_fn_generator.get_input_vocab_size()
        self._params['target_vocab_size'] = self._input_fn_generator.get_target_vocab_size()
        self._model_class = getattr(models, self._flags.model_type)
        # self._graph.info()
    def predict(self):
        commit_id, repos_path = get_commit_id(os.path.realpath(__file__))
        print("source code path:{}\ncommit-id: {}".format(repos_path, commit_id))
        print("tf-version: {}".format(tf.__version__))

        if not self._model:
            self._model = self._model_class(self._params)
        if not self._model.graph_train:
            self._model.graph_train = self._model.get_graph()
            self._model.set_optimizer()
            self._model.set_interface(self._input_fn_generator.get_input_fn_val())
            self._model.graph_train.print_params()
            self._model.graph_train.summary()

        checkpoint_obj = tf.train.Checkpoint(step=self._model.graph_train.global_step, optimizer=self._model.optimizer,
                                             model=self._model.graph_train)

        if tf.train.get_checkpoint_state(flags.FLAGS.checkpoint_dir):
            print("restore from checkpoint: {}".format(flags.FLAGS.checkpoint_dir))
            checkpoint_obj.restore(tf.train.latest_checkpoint(flags.FLAGS.checkpoint_dir))

        #sentence_to_predict="este é um problema que temos que resolver."
        #real_translation="this is a problem we have to solve ."
        #sentence_to_predict="vou então muito rapidamente partilhar convosco algumas histórias de algumas coisas mágicas que aconteceram."
        #real_translation="so i 'll just share with you some stories very quickly of some magical things that have happened ."
        #sentence_to_predict="os meus vizinhos ouviram sobre esta ideia."
        #real_translation="and my neighboring homes heard about this idea ."
        sentence_to_predict="eu dou uma palestra sobre o transformer na citnet."
        real_translation="i give a talk about transformer on citnet ."
        start_token = [self._input_fn_generator._tokenizer_inp.vocab_size]
        end_token = [self._input_fn_generator._tokenizer_inp.vocab_size + 1]

        # inp sentence is portuguese, hence adding the start and end token
        inp_sentence = start_token + self._input_fn_generator._tokenizer_inp.encode(sentence_to_predict) + end_token
        encoder_input = tf.expand_dims(inp_sentence, 0)

        # as the target is english, the first word to the transformer should be the
        # english start token.
        decoder_input = [self._input_fn_generator._tokenizer_tar.vocab_size]
        output = tf.expand_dims(decoder_input, 0)

        for i in range(400):
            inputs={'inputs':encoder_input,'tar_inp':output}
            # predictions.shape == (batch_size, seq_len, vocab_size)
            # predictions, attention_weights
            graph_out = self._model.graph_train(inputs,training=False)
            predictions=graph_out['logits']
            attention_weights=graph_out['attention_weights']

            # select the last word from the seq_len dimension
            predictions = predictions[: ,-1:, :]  # (batch_size, 1, vocab_size)

            predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

            # return the result if the predicted_id is equal to the end token
            if predicted_id == self._input_fn_generator._tokenizer_tar.vocab_size+1:
             break

            # concatentate the predicted_id to the output which is given to the decoder
            # as its input.
            output = tf.concat([output, predicted_id], axis=-1)

        output= tf.squeeze(output, axis=0)
        decoded_input = self._input_fn_generator._tokenizer_inp.decode([i for i in inp_sentence
                                            if i < self._input_fn_generator._tokenizer_inp.vocab_size])
        predicted_sentence = self._input_fn_generator._tokenizer_tar.decode([i for i in output
                                            if i < self._input_fn_generator._tokenizer_tar.vocab_size])

        print('Input: {}'.format(sentence_to_predict))
        print('Predicted translation: {}'.format(predicted_sentence))
        print('Real translation: {}'.format((real_translation)))
        #self.plot_attention_weights(attention_weights, sentence_to_predict, output, 'decoder_layer4_block2')


    def plot_attention_weights(self,attention, sentence, result, layer):
        fig = plt.figure(figsize=(16, 8))

        sentence = self._input_fn_generator._tokenizer_inp.encode(sentence)

        attention = tf.squeeze(attention[layer], axis=0)

        for head in range(attention.shape[0]):
            ax = fig.add_subplot(2, 4, head+1)

            # plot the attention weights
            ax.matshow(attention[head][:-1, :], cmap='viridis')

            fontdict = {'fontsize': 10}

            ax.set_xticks(range(len(sentence)+2))
            ax.set_yticks(range(len(result)))

            ax.set_ylim(len(result)-1.5, -0.5)

            ax.set_xticklabels(
                ['<start>']+[self._input_fn_generator._tokenizer_inp.decode([i]) for i in sentence]+['<end>'],
                fontdict=fontdict, rotation=90)

            ax.set_yticklabels([self._input_fn_generator._tokenizer_tar.decode([i]) for i in result
                                if i < self._input_fn_generator._tokenizer_tar.vocab_size],
                               fontdict=fontdict)

            ax.set_xlabel('Head {}'.format(head+1))

        plt.tight_layout()
        plt.show()




if __name__ == '__main__':
    # logging.basicConfig(level=logging.INFO)
    trainer = TrainerTL()
    if not trainer._flags.predict_mode:
        trainer.train()
    else:
        trainer.predict()
