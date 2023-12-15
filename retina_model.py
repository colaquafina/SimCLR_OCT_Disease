from typing import Tuple

from typing import Callable, Dict
from base_model import Model
import tensorflow as tf
import numpy as np

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