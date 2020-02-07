
class TransformerLoss():

    def __init__(self):
        
        self.loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(
                        from_logits=True, reduction='none')

    def __call__(self, labels, logits, loss_mask):

        losses = self.loss_obj(labels, logits)

        mean_loss = tf.reduce_mean(tf.boolean_mask(losses, loss_mask))

        return mean_loss 


# # Optimizer

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps
    
    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def TransformerOptimizer(d_model):
    
    learning_rate = CustomSchedule(d_model)

    return tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, 
                                     epsilon=1e-9)
