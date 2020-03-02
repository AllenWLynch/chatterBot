


import numpy as np
import tensorflow as tf
import os
import datetime

class ChatBotTrainer():

    def __init__(self, sentencepiece_model, chatbot_model, optimizer, 
        checkpoint_dir = 'checkpoints', log_dir = 'logs', load_from_checkpoint = False, **kwargs):

        self.model = chatbot_model

        self.loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        self.optimizer = optimizer

        self.test_metrics = []
        self.train_metrics = []

        self.decoder = sentencepiece_model

        self.train_steps = 0
        self.eval_steps = 0

        self.logger = tf.summary.create_file_writer(
                os.path.join(
                    log_dir,
                    "fit/",
                    datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                ))

        checkpoint = tf.train.Checkpoint(transformer = self.model)

        if load_from_checkpoint:
            try:
                checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir)).assert_consumed()
            except Exception as err:
                print('Failed to load models from checkpoint!')
                raise err
        
        self.logdir = log_dir

        self.checkpoint_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=5)

    def add_train_metric(self, metric):
        self.train_metrics.append(metric)

    def add_test_metric(self, metric):
        self.test_metrics.append(metric)

    def __call__(self, *args, **kwargs):
        return self.model(args, **kwargs)

    @tf.function()
    def train_step(self, X, Y):

        with tf.GradientTape() as tape:

            logits, loss_weights = self.model(X, training = True)

            loss = self.loss_obj(Y, logits, sample_weight = loss_weights)

            scaled_loss = self.optimizer.get_scaled_loss(loss)

        scaled_gradients = tape.gradient(scaled_loss, self.model.trainable_weights)

        gradients = self.optimizer.get_unscaled_gradients(scaled_gradients)

        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_weights))
        
        for metric in self.train_metrics:
            metric.update_state(Y, logits, sample_weight = loss_weights)

        return loss, gradients

    @tf.function()
    def test_step(self, X, Y):

        logits, loss_weights = self.model(X, training = False)

        for metric in self.test_metrics:
            metric.update_state(Y, logits, sample_weight = loss_weights)

    @tf.function()
    def get_probabilities(self, logits, weights, temperature = 1.0):

        probs = tf.nn.softmax(logits/temperature, axis = -1)

        return probs

    def respond(self, context, sender, author, cutoff = 40, temperature = 1.0):
        
        assert(context.shape[0] == 1), 'Inference only works with batch size of 1'
        
        response_len = context.shape[-1]

        if cutoff is None:
            cutoff = response_len

        response = [[1]]
        idx = 0

        author_id = author[0]

        encoded_context, context_attn_mask = self.model.encode_context(context, sender, training = False)

        for i in range(cutoff):
            
            padded_response = tf.keras.preprocessing.sequence.pad_sequences(response, response_len)
            padded_author = tf.keras.preprocessing.sequence.pad_sequences(author, response_len)
            
            output_logits, loss_weights = self.model.decode_response(
                padded_response, 
                padded_author, 
                encoded_context, context_attn_mask, 
                training = False)

            probs = self.get_probabilities(output_logits, loss_weights, temperature= temperature).numpy()[0,-1]
            
            idx = np.random.choice(len(probs), p = probs)
            response = tf.concat([response, [[idx]]], axis = -1)
            author = tf.concat([author, [author_id]], axis = -1)

            if idx == 2:
                break

        return self.decoder.DecodeIds(response.numpy()[0].tolist())
    
    def train_epoch(self, steps, dataset, log_frequency = 50, debugging = False):

        for i, (X, Y) in enumerate(dataset.take(steps)):

            print('\rStep {}/{}'.format(str(i + 1), str(steps)), end = '')

            loss, grads = self.train_step(X,Y)

            if i % log_frequency == 0:
                with self.logger.as_default():
                     for train_metric in self.train_metrics:
                         tf.summary.scalar(train_metric.name, train_metric.result()/log_frequency, step = self.train_steps)
                         train_metric.reset_states()

                if debugging:
                    weight_norms = [tf.norm(w) for w in self.model.trainable_weights]
                    max_idx = np.argmax(weight_norms)
                    print('Maxnorm: {}, {}'.format(self.model.trainable_weights[max_idx].name, str(weight_norms[max_idx])))


            self.train_steps = self.train_steps + 1
        print('')

    def evaluate(self, steps, dataset):

        examples = []
        for i, (X, Y) in enumerate(dataset.take(steps)):

            print('\rValidation Step {}/{}'.format(str(i+1), str(steps)), end = '')

            self.test_step(X, Y)

        with self.logger.as_default():
            for test_metric in self.test_metrics:
                tf.summary.scalar(test_metric.name, test_metric.result()/steps, step = self.eval_steps)
                test_metric.reset_states()

        print('')
        self.eval_steps = self.eval_steps + 1

    def show_inference_samples(self, dataset, num_samples, max_sample_length, temperature):

        samples = []
        for (X, context) in dataset.take(num_samples):

            prediction = self.respond(*X, max_sample_length, temperature = temperature)
            samples.append(context.numpy()[0].decode() + prediction)

        sample_str = str('\n\n'.join(["__Sample_{}________\n{}".format(str(i + 1), sample) for i, sample in enumerate(samples)]))

        with self.logger.as_default():
            tf.summary.text('Response samples', sample_str, step = self.eval_steps)
        print(sample_str)


    def fit(self, train_dataset, test_dataset, inference_dataset,
                epochs = 100, steps_per_epoch = 10000, evaluation_steps = 100, checkpoint_every = 5,
                logfreq = 50, num_samples = 3, temperature = 0.9, inference_length_cutoff = 30, debugging = False):

        print('Open tensorboard to "{}" to monitor training'.format(self.logdir))
        for epoch in range(epochs):
            print('\nEPOCH ', epoch + 1)
            
            self.train_epoch(steps_per_epoch, train_dataset, logfreq, debugging= debugging)

            self.evaluate(evaluation_steps, test_dataset)

            self.show_inference_samples(inference_dataset, num_samples, inference_length_cutoff, temperature)

            if (epoch + 1) % checkpoint_every == 0:
                self.checkpoint_manager.save()
                print('Saved Checkpoint!')    

        print('Training complete! Saving final model.')
        self.checkpoint_manager.save()
        
    def prompt_for_save(self, checkpoint_manager):
        print('Training interupted!')
        user_input = ''
        while not (user_input == 'y' or user_input == 'n'):
            user_input = input('Save model\'s current state?: [y/n]')
        if user_input == 'y':
            checkpoint_manager.save()
            print('Saved checkpoint!')