import functools
import tensorflow as tf
import numpy as np
from skimage.transform import rotate
import random


CROP_PROPORTION = 0.875  # Standard for ImageNet.


def random_apply(func, p, x):
    """Randomly apply function func to x with probability p."""
    return tf.cond(
        tf.less(
            tf.random.uniform([], minval=0, maxval=1, dtype=tf.float32),
            tf.cast(p, tf.float32)), lambda: func(x), lambda: x)

def random_brightness(image, max_delta):
    """A multiplicative vs additive change of brightness."""
    factor = tf.random.uniform([], tf.maximum(1.0 - max_delta, 0),
                                1.0 + max_delta)
    image = image * factor
    return image

def to_grayscale(image, keep_channels=True):
    image = tf.image.rgb_to_grayscale(image)
    if keep_channels:
        image = tf.tile(image, [1, 1, 3])
    return image

def color_jitter(image, strength, random_order=True):
    """Distorts the color of the image.
  Args:
    image: The input image tensor.
    strength: the floating number for the strength of the color augmentation.
    random_order: A bool, specifying whether to randomize the jittering order.
    impl: 'simclrv1' or 'simclrv2'.  Whether to use simclrv1 or simclrv2's
        version of random brightness.
  Returns:
    The distorted image tensor.
  """
    brightness = 0.8 * strength
    contrast = 0.8 * strength
    saturation = 0.8 * strength
    hue = 0.2 * strength
    if random_order:
        return color_jitter_rand(
            image, brightness, contrast, saturation, hue)
    else:
        return color_jitter_nonrand(
            image, brightness, contrast, saturation, hue)
    

def color_jitter_nonrand(image,
                         brightness=0,
                         contrast=0,
                         saturation=0,
                         hue=0):
    with tf.name_scope('distort_color'):
        def apply_transform(i, x, brightness, contrast, saturation, hue):
            """Apply the i-th transformation."""
            if brightness != 0 and i == 0:
                x = random_brightness(x, max_delta=brightness)
            elif contrast != 0 and i == 1:
                x = tf.image.random_contrast(
                    x, lower=1 - contrast, upper=1 + contrast)
            elif saturation != 0 and i == 2:
                x = tf.image.random_saturation(
                    x, lower=1 - saturation, upper=1 + saturation)
            elif hue != 0:
                x = tf.image.random_hue(x, max_delta=hue)
            return x
        
        for i in range(4):
            image = apply_transform(i, image, brightness, contrast, saturation, hue)
            # image = tf.clip_by_value(image, 0., 1.)
        return image
    
def color_jitter_rand(image,
                      brightness=0,
                      contrast=0,
                      saturation=0,
                      hue=0):
    with tf.name_scope('distort_color'):
        def apply_transform(i, x):
            """Apply the i-th transformation."""

            def brightness_foo():
                if brightness == 0:
                    return x
                else:
                    return random_brightness(x, max_delta=brightness)

            def contrast_foo():
                if contrast == 0:
                    return x
                else:
                    return tf.image.random_contrast(x, lower=1 - contrast, upper=1 + contrast)

            def saturation_foo():
                if saturation == 0:
                    return x
                else:
                    return tf.image.random_saturation(
                        x, lower=1 - saturation, upper=1 + saturation)

            def hue_foo():
                if hue == 0:
                    return x
                else:
                    return tf.image.random_hue(x, max_delta=hue)

            x = tf.cond(tf.less(i, 2),
                        lambda: tf.cond(tf.less(i, 1), brightness_foo, contrast_foo),
                        lambda: tf.cond(tf.less(i, 3), saturation_foo, hue_foo))
            return x

        perm = tf.random.shuffle(tf.range(4))
        for i in range(4):
            image = apply_transform(perm[i], image)
        return image
    
def _compute_crop_shape(
        image_height, image_width, aspect_ratio, crop_proportion):

    image_width_float = tf.cast(image_width, tf.float32)
    image_height_float = tf.cast(image_height, tf.float32)

    def _requested_aspect_ratio_wider_than_image():
        crop_height = tf.cast(
            tf.math.rint(crop_proportion / aspect_ratio * image_width_float),
            tf.int32)
        crop_width = tf.cast(
            tf.math.rint(crop_proportion * image_width_float), tf.int32)
        return crop_height, crop_width
    def _image_wider_than_requested_aspect_ratio():
        crop_height = tf.cast(
            tf.math.rint(crop_proportion * image_height_float), tf.int32)
        crop_width = tf.cast(
            tf.math.rint(crop_proportion * aspect_ratio * image_height_float),
            tf.int32)
        return crop_height, crop_width

    return tf.cond(
        aspect_ratio > image_width_float / image_height_float,
        _requested_aspect_ratio_wider_than_image,
        _image_wider_than_requested_aspect_ratio)

def center_crop(image, height, width, crop_proportion):

    shape = tf.shape(image)
    image_height = shape[0]
    image_width = shape[1]
    crop_height, crop_width = _compute_crop_shape(
        image_height, image_width, height / width, crop_proportion)
    offset_height = ((image_height - crop_height) + 1) // 2
    offset_width = ((image_width - crop_width) + 1) // 2
    image = tf.image.crop_to_bounding_box(
        image, offset_height, offset_width, crop_height, crop_width)

    image = tf.image.crop_to_bounding_box(
        image, offset_height, offset_width, crop_height, crop_width)

    image = tf.image.resize([image], [height, width],
                            method=tf.image.ResizeMethod.BICUBIC)[0]

    return image

def distorted_bounding_box_crop(image,
                                bbox,
                                min_object_covered=0.1,
                                aspect_ratio_range=(0.75, 1.33),
                                area_range=(0.05, 1.0),
                                max_attempts=100,
                                scope=None):
    with tf.name_scope(scope or 'distorted_bounding_box_crop'):
        shape = tf.shape(image)
        sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
            shape,
            bounding_boxes=bbox,
            min_object_covered=min_object_covered,
            aspect_ratio_range=aspect_ratio_range,
            area_range=area_range,
            max_attempts=max_attempts,
            use_image_if_no_bounding_boxes=True)
        bbox_begin, bbox_size, _ = sample_distorted_bounding_box

        # Crop the image to the specified bounding box.
        offset_y, offset_x, _ = tf.unstack(bbox_begin)
        target_height, target_width, _ = tf.unstack(bbox_size)
        image = tf.image.crop_to_bounding_box(
            image, offset_y, offset_x, target_height, target_width)

        return image
                  
def crop_and_resize(image, height, width):

    bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
    aspect_ratio = width / height
    image = distorted_bounding_box_crop(
        image,
        bbox,
        min_object_covered=0.1,
        # aspect_ratio_range=(3. / 4 * aspect_ratio, 4. / 3. * aspect_ratio),
        aspect_ratio_range=(0.8 * aspect_ratio, 1.2 * aspect_ratio),
        area_range=(0.5, 1.0),
        max_attempts=100,
        scope=None)
    return tf.image.resize([image], [height, width],
                           method=tf.image.ResizeMethod.BICUBIC)[0]

def gaussian_blur(image, kernel_size, sigma, padding='SAME'):

    radius = tf.cast(kernel_size / 2, dtype=tf.int32)
    kernel_size = radius * 2 + 1
    x = tf.cast(tf.range(-radius, radius + 1), dtype=tf.float32)
    blur_filter = tf.exp(-tf.pow(x, 2.0) /
                         (2.0 * tf.pow(tf.cast(sigma, dtype=tf.float32), 2.0)))
    blur_filter /= tf.reduce_sum(blur_filter)
    # One vertical and one horizontal filter.
    blur_v = tf.reshape(blur_filter, [kernel_size, 1, 1, 1])
    blur_h = tf.reshape(blur_filter, [1, kernel_size, 1, 1])
    num_channels = tf.shape(image)[-1]
    blur_h = tf.tile(blur_h, [1, 1, num_channels, 1])
    blur_v = tf.tile(blur_v, [1, 1, num_channels, 1])
    expand_batch_dim = image.shape.ndims == 3
    if expand_batch_dim:
        # Tensorflow requires batched input to convolutions, which we can fake with
        # an extra dimension.
        image = tf.expand_dims(image, axis=0)
    blurred = tf.nn.depthwise_conv2d(
        image, blur_h, strides=[1, 1, 1, 1], padding=padding)
    blurred = tf.nn.depthwise_conv2d(
        blurred, blur_v, strides=[1, 1, 1, 1], padding=padding)
    if expand_batch_dim:
        blurred = tf.squeeze(blurred, axis=0)
    return blurred

def random_crop_with_resize(image, height, width, p=1.0):
    def _transform(image):  # pylint: disable=missing-docstring
        image = crop_and_resize(image, height, width)
        return image

    return random_apply(_transform, p=p, x=image)

def random_color_jitter(image, p=1.0, color_jitter_strength=0.5):
    def _transform(image):
        color_jitter_t = functools.partial(
            color_jitter, strength=color_jitter_strength)
        image = random_apply(color_jitter_t, p=0.8, x=image)
        return random_apply(to_grayscale, p=0.2, x=image)

    return random_apply(_transform, p=p, x=image)

def random_blur(image, height, width, p=1.0):
    del width

    def _transform(image):
        sigma = tf.random.uniform([], 0.1, 2.0, dtype=tf.float32)
        return gaussian_blur(
            image, kernel_size=height // 10, sigma=sigma, padding='SAME')

    return random_apply(_transform, p=p, x=image)

def batch_random_blur(images_list, height, width, blur_probability=0.5):
    def generate_selector(p, bsz):
        shape = [bsz, 1, 1, 1]
        selector = tf.cast(
            tf.less(tf.random.uniform(shape, 0, 1, dtype=tf.float32), p),
            tf.float32)
        return selector

    new_images_list = []
    for images in images_list:
        images_new = random_blur(images, height, width, p=1.)
        selector = generate_selector(blur_probability, tf.shape(images)[0])
        images = images_new * selector + images * (1 - selector)
        images = tf.clip_by_value(images, 0., 1.)
        new_images_list.append(images)

    return new_images_list

def preprocess_for_train(image,
                         height,
                         width,
                         color_distort=True,
                         crop=True,
                         flip=True,
                         color_jitter_strength=0.9):
    if crop:
        image = random_crop_with_resize(image, height, width)
    if flip:
        image = tf.image.random_flip_left_right(image)
    if color_distort:
        image = random_color_jitter(image,
                                    color_jitter_strength=color_jitter_strength)
    image = tf.reshape(image, [height, width, 3])
    image = tf.clip_by_value(image, 0., 1.)
    return image

def preprocess_for_eval(image, height, width, crop=True):
    if crop:
        image = center_crop(image, height, width, crop_proportion=CROP_PROPORTION)
    image = tf.reshape(image, [height, width, 3])
    image = tf.clip_by_value(image, 0., 1.)
    return image

def preprocess_image(image, height, width, is_training=False,
                     color_distort=True, test_crop=True):
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    if is_training:
        return preprocess_for_train(image, height, width, color_distort)
    else:
        return preprocess_for_eval(image, height, width, test_crop)
    
def train_classification_aug(img, lb,
                             img_size=128,
                             brightness_delta=0.8,
                             contrast_delta=0.5,
                             saturation_delta=0.5,
                             hue_delta=0.2,
                             max_rot_angle=45.):
    img = tf.cast(img, dtype=tf.float32) / 255.
    IMG_SIZE = img_size

    angle_rad = max_rot_angle / 180. * np.pi

    padding = IMG_SIZE // 4
    precrop_shape = IMG_SIZE + padding

    img = tf.image.resize(img, (precrop_shape, precrop_shape))
    img = tf.image.random_crop(img, (IMG_SIZE, IMG_SIZE, 3))
    img = tf.image.random_brightness(img, brightness_delta)
    img = tf.image.random_contrast(img, 1 - contrast_delta, 1 + contrast_delta)
    img = tf.image.random_saturation(img, 1 - saturation_delta, 1 + saturation_delta)
    img = tf.image.random_hue(img, hue_delta)

    img = tf.image.random_flip_left_right(img)
    img = rotate(img, random.uniform(-angle_rad, angle_rad), mode = 'edge')

    img = tf.clip_by_value(img, 0., 1.0)
    return img, lb  # tf.one_hot(lb, 4, )
