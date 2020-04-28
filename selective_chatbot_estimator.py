

#%%
import numpy as np
import tensorflow as tf
import os
import datetime
#%%
class ChatBotTrainer():

    def __init__(self, chatbot_model, optimizer, 
        checkpoint_dir = 'checkpoints', log_dir = 'logs', load_from_checkpoint = False, **kwargs):

        self.model = chatbot_model

        self.optimizer = optimizer

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
                print("Restored from checkpoint")
            except Exception as err:
                print('Failed to load models from checkpoint!')
                raise err
        
        self.logdir = log_dir

        self.checkpoint_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=5)

        self.loss_tracker = tf.keras.metrics.Mean()

    def __call__(self, *args, **kwargs):
        return self.model(args, **kwargs)

    @tf.function()
    def train_step(self, X, margin):

        with tf.GradientTape() as tape:

            compatibilities = self.model(X)

            loss = self.model.online_batchall_triplet_loss(compatibilities, margin)

            scaled_loss = self.optimizer.get_scaled_loss(loss)

        scaled_gradients = tape.gradient(scaled_loss, self.model.trainable_weights)

        gradients = self.optimizer.get_unscaled_gradients(scaled_gradients)

        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_weights))
        
        return loss, gradients

    @tf.function()
    def test_step(self, X, margin):

        compatibilities = self.model(X, training = False)

        loss = self.model.online_batchall_triplet_loss(compatibilities, margin)

        return loss
    
    def train_epoch(self, steps, dataset, log_frequency = 50, debugging = False):

        for i, (X, Y) in enumerate(dataset.take(steps)):

            print('\rStep {}/{}'.format(str(i + 1), str(steps)), end = '')

            loss, grads = self.train_step(X,Y)

            self.loss_tracker.update_state([loss])

            if i % log_frequency == 0:
                with self.logger.as_default():
                    tf.summary.scalar('Training Triplet Loss', self.loss_tracker.result(), step = self.train_steps)
                    self.loss_tracker.reset_states()

                if debugging:
                    weight_norms = [tf.norm(w) for w in self.model.trainable_weights]
                    max_idx = np.argmax(weight_norms)
                    print('Maxnorm: {}, {}'.format(self.model.trainable_weights[max_idx].name, str(weight_norms[max_idx])))

            self.train_steps = self.train_steps + 1
        print('')

    def evaluate(self, steps, dataset):

        examples = []
        for i, X in enumerate(dataset.take(steps)):

            print('\rValidation Step {}/{}'.format(str(i+1), str(steps)), end = '')

            loss = self.test_step(X)

            self.loss_tracker.update_state([loss])

        with self.logger.as_default():
            tf.summary.scalar('Test Triplet Loss', self.loss_tracker.result(), step = self.eval_steps)
            self.loss_tracker.reset_states()

        print('')
        self.eval_steps = self.eval_steps + 1


    def fit(self, train_dataset, test_dataset, inference_dataset,
                epochs = 100, steps_per_epoch = 10000, evaluation_steps = 100, checkpoint_every = 5,
                logfreq = 50, debugging = False):

        print('Open tensorboard to "{}" to monitor training'.format(self.logdir))
        for epoch in range(epochs):
            print('\nEPOCH ', epoch + 1)
            
            self.train_epoch(steps_per_epoch, train_dataset, logfreq, debugging= debugging)

            self.evaluate(evaluation_steps, test_dataset)

            if (epoch + 1) % checkpoint_every == 0:
                self.checkpoint_manager.save()
                print('Saved Checkpoint!')    

        print('Training complete! Saving final model.')
        self.checkpoint_manager.save()
        
    def prompt_for_save(self):
        print('Training interupted!')
        user_input = ''
        while not (user_input == 'y' or user_input == 'n'):
            user_input = input('Save model\'s current state?: [y/n]')
        if user_input == 'y':
            self.checkpoint_manager.save()
            print('Saved checkpoint!')