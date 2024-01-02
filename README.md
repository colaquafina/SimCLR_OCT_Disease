# Deep learning model to classify the OCT Disease

The dataset in kaggle (https://www.kaggle.com/datasets/paultimothymooney/kermany2018) is an OCT dataset organized into 3 folders (train, test, val) and contains subfolders for each image category (NORMAL,CNV,DME,DRUSEN). I learn a lot from this github page (https://github.com/anoopsanka/retinal_oct). In this project, it provides two models to do the classification tasks: ResNet (https://arxiv.org/abs/1512.03385) and SimCLR (https://arxiv.org/abs/2002.05709)

## ResNet
The key innovation of ResNet is the use of **skip connections** or residual connections that allow the network to learn residual functions with reference to the layer inputs. In other words, instead of trying to learn the underlying mapping between the input and output directly, ResNet allows the network to fit the residual mapping. The residual connections enable the network to learn the identity function, which helps to mitigate the vanishing gradient problem that can occur in very deep networks.

If we use $F(x)$ to represent the CNN, then the output is 
$$
y=F(x)+x
$$
By calculating the backpropagations of ResNet, we can find that the total gradient is
![image](https://github.com/colaquafina/ResNet_OCT_Disease/assets/86960905/a50394f7-4123-4045-8dbd-66cfca71fa8b)

In the functions $\xi$ is the loss function, $xl$ is the parameters of layers l, and $xL$ is the parameters of the last layer. It means the **gradient would not vanish to 0** because of the short cut connection.

In the ResNet, there are two kinds of Resnet blocks, **Residual Block and Bottleneck Block**. For the former one, it has 2 3\*3 convolution layers with connections and for the last one, it has 1\*1, 3\*3 and 1\*1 convolution layers with connections. In different ResNet models, it would have differnet number of Resnet blocks and different kinds of Resnet blocks. For example, ResNet-34 has 34 Residual Blocks, and ResNet-50 has 50 Bottleneck Blocks.

In this project, we build a **base_model** class (base_model.py) and use the **ResNet50V2** (retina_model.py) from tensorflow to instantiate the ResNet model. Also, you can use other versions of ResNet to build the model.

## Unsupervised learning
In deep learning, the model trained on data with label is supervised learning. However, it is hard to get a lot of labeled data for training, so we need unsupervised learning, which is training on unlabeled data. After that, the trained model can be used in the downstream tasked by using the labeled data to train the trained model. It can increase the accuracy. There are many methods, such as the model we used today, SimCLR.

The SimCLR uses contrast learning to do the classification tasks. The contrast learning is based on the theory that the similar pictures would have shorter diatance between each other than other pictures. Therefore, in the model, it uses data augmentations to process the pictures into two versions, and then feed the pictures to the based model and a projection layer, and then minimize the distance between the two pictures.
![image](https://github.com/colaquafina/ResNet_OCT_Disease/assets/86960905/f4042d2e-c9a9-4e37-a065-e80207163a36)

In this picture, $f(·)$ is the based model (ResNet is used in here), the $g(·)$ is the projection layer. By using the projection layers, the model performance could be much better.

## Data processing
In this project, we use the OCT data from kaggle, which inlcudes the training and testing data. We use the **RetianDataset** class from `create_dataset.py` inherent from `tfds.core.GeneratorBasedBuilder`. This is a class for people to build tf dataset from different kinds of data. It has three functions. `_info()` defines the dataset information (`tfds.core.DatasetInfo`), including the dataset builder, driscription and data features. `_split_generators()` can download the data by using `dl_manager.download` and use the ` tfds.core.SplitGenerator` to split the data into training and testing data. `_generate_examples` can be used to yield examples into dataset. In this way, we build a dataset called RetinaDataset. 

Then we need to preprocess the images (`data_process.py`). For the ResNet, the image processing can be used to increase the number of training images can increase the model generalizations. There are some basic image processing methods, including rotation, normalization and crop. However, for the SimCLR, the data augmentations is used to produce two versions of images. In the paper, it can crop, resize, rotate the images, and it would distort the color of the images and blur the images. Also it is proved that using the blur in preprocessing can improve the model performance.

And then we load and process the dataset (`dataset_retina.py`)

## ResNet Model
Here, to make it convenient to use the ResNet model, we build a base model (`base_model.py`). Base model is a model container. We can put a ResNet50V2 (`retina_model.py`) in it and the dataset. Then we can use the model to train the data.

## SimCLR Model.
SimCLR is comsisted with a ResNet and a projections layer. However, with the development of the deep learning, there are many skills to improve the model performance, so it is more convenient to write the layers inherent from the tensorflow layers.

In `layers.py`, it has (1) **BatchNormRelu**. It is consisted with a BatchNormalization layer and a Relu layer. Batch normalization is typically applied after the convolutional or fully connected layers and before the activation function. It can speed up the training process and smoothens the loss function. (2) **Conv2dFixedPadding**. It would pad the input at first and then pass it throught the Conv2D. (3) **IdentityLayer**. It returns the same output as the layer input. (4) **SE_Layer** (https://arxiv.org/abs/1709.01507). SE layers is called "Squeeze-and-Excitation block" (SE block). It incorporates the channel relationship of the data into training. For the SE block, the input ($H\times W\times C$) is *squeezed* by using a global averaging pooling to generate a channel-wise stastics ($1\times 1\times C$). In the *excitation*, it used to FC layers with activation function (relu and sigmoid seperately) to activate the channle-wise stastics. Finally, it multiplicates channel-wise with the input. SE block can be added after the Conv2D, and by stacking the SE blocks and Conv2D blocks, we can get SE Net. 
![image](https://github.com/colaquafina/ResNet_OCT_Disease/assets/86960905/7013a425-8dd6-4883-8f27-c5da34c2bce1)

![image](https://github.com/colaquafina/ResNet_OCT_Disease/assets/86960905/e7e4ce49-4841-455e-a3ef-c543e340094c)

Therefore, from the **SE_Layer** in the `layers.py`, we can find that the input is averaged across the channels, and then after two convolution layers with activation layers, the short cut output times the input data and produce the output.
