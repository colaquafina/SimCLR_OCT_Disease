import tensorflow as tf
import math

class WarmUpAndCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Applies a warmup schedule on a given learning rate decay schedule."""
    """
    Warm up is a method to prevent from missing the optimized value. It is would increase the 
    learning rate slowly from small to original learning in the beginning of the training (warm up steps)
    
    The cosine decay is a learning rate schedule that reduces the learning rate in a 
    smooth and gradual manner 1. The learning rate is decreased following a cosine function, 
    which starts at the initial learning rate and decays to a minimum value over a set number of epochs 2.
    The cosine decay is provided by tensorflow.
    """
    def __init__(self, base_learning_rate, num_examples,
                 name=None,
                 train_epochs=1000,
                 train_batch_size=256,
                 learning_rate_scaling='linear',
                 warmup_epochs=10,
                 ):
        super(WarmUpAndCosineDecay, self).__init__()
        self.base_learning_rate = base_learning_rate
        self.num_examples = num_examples
        self._name = name
        self.learning_rate_scaling = learning_rate_scaling
        self.train_batch_size = train_batch_size
        self.warmup_epochs = warmup_epochs
        self.train_epochs = train_epochs

    def __call__(self, step):
        with tf.name_scope(self._name or 'WarmUpAndCosineDecay'):
            warmup_steps = int(
                round(self.warmup_epochs * self.num_examples //
                      self.train_batch_size))
            if self.learning_rate_scaling == 'linear':
                scaled_lr = self.base_learning_rate * self.train_batch_size / 256.
            elif self.learning_rate_scaling == 'sqrt':
                scaled_lr = self.base_learning_rate * math.sqrt(self.train_batch_size)
            else:
                raise ValueError('Unknown learning rate scaling {}'.format(
                    self.learning_rate_scaling))
            learning_rate = (
                step / float(warmup_steps) * scaled_lr if warmup_steps else scaled_lr)

            # Cosine decay learning rate schedule
            total_steps = self.num_examples * self.train_epochs // self.train_batch_size + 1
            # TODO(srbs): Cache this object.
            cosine_decay = tf.keras.experimental.CosineDecay(
                scaled_lr, total_steps - warmup_steps)
            learning_rate = tf.where(step < warmup_steps, learning_rate,
                                     cosine_decay(step - warmup_steps))

            return learning_rate

    def get_config(self):
        return {
            'base_learning_rate': self.base_learning_rate,
            'num_examples': self.num_examples,
        }