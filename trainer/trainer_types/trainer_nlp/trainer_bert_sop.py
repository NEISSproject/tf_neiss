from trainer.trainer_base import TrainerBase
import model_fn.model_fn_nlp.model_fn_bert_sop as models
import util.flags as flags
from input_fn.input_fn_nlp.input_fn_bert_sop import InputFnBertSOP
import os
import time

import tensorflow as tf
from util.misc import get_commit_id

# Model parameter
# ===============
flags.define_string('model_type', 'ModelBertSOP', 'Model Type to use choose from: ModelTriangle')
flags.define_string('tokenizer', "../../../data/tokenizer/tokenizer_de", 'path to subword tokenizer')
flags.define_string('graph', 'BERTMiniSOP', 'class name of graph architecture')
flags.define_list('add_types', str, 'types that are add features int or float',
                  "", [])
flags.define_integer('buffer', 1,
                     'number of training samples hold in the cache. (effects shuffling)')
flags.define_integer('sum_train_batches', 1,
                     'use the average of the gradients sum_train_batches to apply the gradients')
flags.define_integer('max_token_text_part', 80,
                     'maximal number of words in a text part of the input function')
flags.define_boolean('segment_train', False,'If and only if True the training is done with text segments and not with sentences')
flags.define_boolean('predict_mode', False, 'If and only if true the prediction will be accomplished')
flags.define_string('bert_checkpoint_dir', '', 'Checkpoint to save pure bert model information in.')
flags.FLAGS.parse_flags()


class TrainerTFBertSOP(TrainerBase):
    def __init__(self):
        super(TrainerTFBertSOP, self).__init__()
        self._input_fn_generator = InputFnBertSOP(self._flags)
        self._input_fn_generator.print_params()
        self._params['tok_size'] = self._input_fn_generator._tok_vocab_size+3
        self._model_class = getattr(models, self._flags.model_type)
        # self._graph.info()

    def train_sum_batches(self):
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
        checkpoint_manager = tf.train.CheckpointManager(checkpoint=checkpoint_obj, directory=self._flags.checkpoint_dir,
                                                        max_to_keep=1)

        if tf.train.get_checkpoint_state(self._flags.checkpoint_dir):
            print("restore from checkpoint: {}".format(self._flags.checkpoint_dir))
            checkpoint_obj.restore(tf.train.latest_checkpoint(self._flags.checkpoint_dir))
        if self._model.graph_train.global_epoch.numpy() >= self._flags.epochs:
            print('Loaded model already in epoch {}. Evaluation...'.format(
                self._model.graph_train.global_epoch.numpy()))
            self.eval()  # run eval() if epochs reach on first attempt
            self.export()
            return 0
        else:
            print('starting in epoch ' + str(self._model.graph_train.global_epoch.numpy()))

        if not self._train_dataset:
            self._train_dataset = self._input_fn_generator.get_input_fn_train()

        train_step_signature = self._model.get_call_graph_signature()

        @tf.function(input_signature=train_step_signature)
        def _train_step_intern(input_features_, targets_):
            with tf.GradientTape() as self.tape:
                self._model.graph_train._graph_out = self._model.graph_train(input_features_, training=True)
                loss = self._model.loss(predictions=self._model.graph_train._graph_out, targets=targets_)
                gradients = self.tape.gradient(loss, self._model.graph_train.trainable_variables)
                #self._model.optimizer.apply_gradients(zip(gradients, self._model.graph_train.trainable_variables))
                #self._model.graph_train.global_step.assign(self._model.optimizer.iterations)
                self._model.graph_train._graph_out["loss"] = tf.reduce_mean(loss)
            return self._model.graph_train._graph_out,gradients

        while True:
            if self._model.graph_train.global_epoch.numpy() >= self._flags.epochs:
                break
            self.epoch_loss = 0.0
            t1 = time.time()
            self._model.set_mode("train")
            train_batch_number = 0
            i=0
            for (batch, (input_features, targets)) in enumerate(self._input_fn_generator.get_input_fn_train()):
                # do the _train_step as tf.function to improve performance
                train_out_dict,cur_gradients = _train_step_intern(input_features, targets)
                if i==0:
                    gradients=cur_gradients
                else:
                    gradients+=cur_gradients
                i+=1
                if i%self._flags.sum_train_batches==0:
                    i=0
                    gradients=[tf.math.scalar_mul(1.0/float(self._flags.sum_train_batches),gradient) for gradient in gradients]
                    self._model.optimizer.apply_gradients(zip(gradients, self._model.graph_train.trainable_variables))
                    self._model.graph_train.global_step.assign(self._model.optimizer.iterations)

                self._model.to_tensorboard(train_out_dict, targets, input_features)
                self.epoch_loss += train_out_dict["loss"]
                train_batch_number = batch
                if batch + 1 >= int(self._flags.samples_per_epoch / self._flags.train_batch_size):
                    # stop endless '.repeat()' dataset with break
                    break

            self.epoch_loss /= float(train_batch_number + 1.0)
            self._model.graph_train.global_epoch.assign_add(1)
            print("\nEPOCH:   {:10.0f}, optimizer steps: {:9}".format(self._model.graph_train.global_epoch.numpy(),
                                                                      self._model.graph_train.global_step.numpy()))
            print("train-loss:{:8.3f}, samples/seconde:{:8.1f}, time:{:6.1f}"
                  .format(self.epoch_loss, self._flags.samples_per_epoch / (time.time() - t1), time.time() - t1))
            # Save checkpoint each epoch
            checkpoint_manager.save()
            self._model.write_tensorboard()

            # Evaluation on this checkpoint
            self._model.set_mode("eval")
            self.eval()
            self._model.write_tensorboard()

        self.export()

    def save_bert(self):
        bert_checkpoint_obj = tf.train.Checkpoint(step=self._model.graph_train.global_step, optimizer=self._model.optimizer,
                                             model=self._model.graph_train.bert)
        bert_checkpoint_manager = tf.train.CheckpointManager(checkpoint=bert_checkpoint_obj, directory=self._flags.bert_checkpoint_dir,
                                                        max_to_keep=1)
        bert_checkpoint_manager.save()



if __name__ == '__main__':
    # logging.basicConfig(level=logging.INFO)
    trainer = TrainerTFBertSOP()
    if trainer._flags.sum_train_batches==1:
        trainer.train()
    else:
        trainer.train_sum_batches()
    trainer.save_bert()
