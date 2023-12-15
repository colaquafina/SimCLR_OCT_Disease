from pathlib import Path
from typing import Callable, Dict

import tensorflow as tf
import numpy as np


WEIGHTS_DIRNAME = Path(__file__).parents[1].resolve() / "weights"
_SEED=42

class Model:

    def __init__(
        self,
        dataset_cls: type,
        network_fn: Callable[..., tf.keras.Model],
        dataset_args: Dict = {}
    ):
        tf.random.set_seed(_SEED)
        np.random.seed(_SEED)

        self.name = f'{self.__class__.__name__}_{dataset_cls.__name__}_{network_fn.__name__}'

        self.data = dataset_cls(**dataset_args)

        self.network = network_fn(input_shape=self.data.input_shape, output_shape=self.data.num_classes)

    def weights_filename(self) -> str:
        WEIGHTS_DIRNAME.mkdir(parents=True, exist_ok=True)
        return str(WEIGHTS_DIRNAME / f"{self.name}_weights.h5")
    
    def image_shape(self):
        return self.data.input_shape

    def fit(self, dataset, batch_size: int = 32, epochs: int = 10, lr: float = 1e-3, verbose: int = 1, callbacks: list = []):
        dataset.train, dataset.validation , dataset.test = dataset.prepare()
        self.network.compile(loss= tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=tf.keras.optimizers.Adam(learning_rate=lr), metrics=['accuracy'])

        class_weight = dataset.get_class_weights()

        fit_kwargs = dict(
                        epochs = epochs,
                        validation_data = dataset.validation.batch(batch_size),
                        verbose = verbose,
                        callbacks = callbacks,
                        class_weight = class_weight
                    )
        self.network.fit(dataset.train.batch(batch_size), **fit_kwargs)

    def evaluate(self, data, batch_size: int = 32, verbose=1):
        return self.network.evaluate(data.batch(batch_size), verbose=verbose)


    def load_weights(self):
        self.network.load_weights(self.weights_filename)

    def save_weights(self):
        self.network.save_weights(self.weights_filename)


from typing import Tuple
def resnetconv(input_shape: Tuple[int, ...], output_shape: Tuple[int, ...]) -> tf.keras.models.Model:
    num_classes = output_shape[0]
    preprocess = tf.keras.applications.resnet_v2.preprocess_input
    base_model = tf.keras.applications.ResNet50V2(include_top=False, weights='imagenet')
    base_model.trainable = False

    inputs = tf.keras.layers.Input(input_shape)
    pool = tf.keras.layers.GlobalAveragePooling2D()
    flatten = tf.keras.layers.Flatten()
    softmax = tf.keras.layers.Dense(num_classes, activation='softmax')

    x = inputs
    x = preprocess(x)
    x = base_model(x)
    x = tf.keras.layers.Conv2D(64, 3, activation="relu")(x)
    x = pool(x)
    x = flatten(x)
    out = softmax(x)

    return tf.keras.Model(inputs=inputs, outputs=out)

from dataset.create_dataset import RetinaDataset

class RetinaModel(Model):


    def __init__(
        self,
        dataset_cls: type = RetinaDataset,
        network_fn: Callable = resnetconv,
        dataset_args: Dict = {}
    ):
        print (dataset_cls)
        super().__init__(dataset_cls, network_fn, dataset_args)


## I think I can use simple way to write this model
# preprocess = tf.keras.applications.resnet_v2.preprocess_input
# base_model = tf.keras.applications.ResNet50V2(include_top=False, weights='imagenet')
# base_model.trainable = False
# model = tf.keras.Sequention([
#     preprocess,
#     base_model,
#     tf.keras.layers.Conv2D(64, 3, activation="relu"),
#     tf.keras.layers.GlobalAveragePooling2D(),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(4, activation='softmax')
# ])

# model.compile(
#     optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3),
#     loss=tf.keras.losses.SparseCategoricalCrossentropy(),
#     metrics=['accuracy']
# )
# fit_kwargs = dict(
#                 epochs = 10,
#                 validation_data = RetinaDataset.validation.batch(32),
#                 verbose = 1,
#                 callbacks = [],
#                 class_weight = RetinaDataset.get_class_weights()
#             )
# model.fit(RetinaDataset.train.batch(32), **fit_kwargs)
