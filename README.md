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
![image](https://github.com/colaquafina/ResNet_OCT_Disease/assets/86960905/fe441af2-5f4e-410a-b46b-9c8bbd8bd4a3)


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
### layers
In `layers.py`, it has (1) **BatchNormRelu**. It is consisted with a BatchNormalization layer and a Relu layer. Batch normalization is typically applied after the convolutional or fully connected layers and before the activation function. It can speed up the training process and smoothens the loss function. 

(2) **Conv2dFixedPadding**. It would pad the input at first and then pass it throught the Conv2D. (3) **IdentityLayer**. It returns the same output as the layer input.

(4) **SE_Layer** (https://arxiv.org/abs/1709.01507). SE layers is called "Squeeze-and-Excitation block" (SE block). It incorporates the channel relationship of the data into training. For the SE block, the input ($H\times W\times C$) is *squeezed* by using a global averaging pooling to generate a channel-wise stastics ($1\times 1\times C$). In the *excitation*, it used to FC layers with activation function (relu and sigmoid seperately) to activate the channle-wise stastics. Finally, it multiplicates channel-wise with the input. SE block can be added after the Conv2D, and by stacking the SE blocks and Conv2D blocks, we can get SE Net. 
![image](https://github.com/colaquafina/ResNet_OCT_Disease/assets/86960905/7013a425-8dd6-4883-8f27-c5da34c2bce1)

![image](https://github.com/colaquafina/ResNet_OCT_Disease/assets/86960905/e7e4ce49-4841-455e-a3ef-c543e340094c)

Therefore, from the **SE_Layer** in the `layers.py`, we can find that the input is averaged across the channels, and then after two convolution layers with activation layers, the short cut output times the input data and produce the output.

(5) **SK_layer** (https://arxiv.org/abs/1903.06586). How to define the size of the receptive field (kernel size) would affect the model performance. In human brain, the receptive field would changes with the contrast of the input, but in convolution models, the kernel size would nuot change. Therefore, the SK unit consists of multiple branches with different kernel sizes that are fused using softmax attention guided by the information in these branches. Different attentions on these branches yield different sizes of the effective receptive fields of neurons in the fusion layer. Multiple SK units are stacked to a deep network termed Selective Kernel Networks (SKNets). 

The SK block can be divided into three steps, split, fuse and select. For the splitting, it used two kernels with different sizes ($3 \times 3\ and\ 5\times 5$) to do the convolution with input ($H' \times W' \times C'$) and produce two blocks $U1 and U2$ ($H \times W \times C$). Also, the larger kernel is dilated from the smaller kernel by dilated convolution, so it would not increase the number of the parameters. In the fuse part, we fuse results from two branches via an element-wise summation and get matrix $U$. We use the globale averaging and FC layer with BatchNormalizationRelu on matrix $U$ to produce matrix $Z$. Finally, in the selecting part, the softmax operator is applied to Z in channle wise (the attention mechanism) to produce the attention weights. After the miltiplication between the attention weights and $U1\ and\ U2$
![image](https://github.com/colaquafina/ResNet_OCT_Disease/assets/86960905/68333831-5ba2-40db-b64e-da3fe7e2994c)

In the code, we use a kernel with size in $3 \times 3$ and twice filter to do the convolution, and then use `tf.split` to splite the output into two matrix (split). Then in the fusing part, we use  `Conv2D_0` and BathNormalizationRelu to produce matrix Z. Later, we use `conv2D_1` with twice filters and softmax operater to procduce the attention weights. 

### Resnet_block
As we said before, the ResNet can be based on **Residual Blcok** and **Bottleneck block**. We write this two blocks in `resnet-block.py` and we change it a little bit to improve the model performance. In the Residual Block, we incorporate the `sk_ratio` and `se_ration` to determine if we need the SK blcok and SE block. Also, the `sk_ratio` is relevant about the use of ResNet-D (I dont know why). Insetad of using a short cut without changing input, it uses an average pool and a conv2d layer to process the input. In conclusion, the new **Residual Block** is shown below.
![image](https://github.com/colaquafina/ResNet_OCT_Disease/assets/86960905/41a498e6-be18-4682-839e-c6a10bc4e9d7)

For the Bottleneck block, we also have similar changes. It is shown below
![image](https://github.com/colaquafina/ResNet_OCT_Disease/assets/86960905/ae01c7ea-133d-4f66-938c-b5cf028b4276)


### Projection_layers
In SimCLR, projection layers is a fully connected layer used to extrace and compress the output of the ResNet. It is proved that calculating the loss function based on the output of the projection layers can get better model performance.

In `projection_layers.py`, the class `ProjectionHead` and `SupervisedHead` inherited from `tf.layers` are based on `LinearLayer` or nonlinear layers. The supervisedhead is used in to downstreat tasked with labeled data. For the `LinearLayer`, it is a fully connected layer (dense layer) with (or without) BatchNormalization layer and bias term. For the `ProjectionHead`, if it is linear, the projectionHead is a `LinearLayer` without bias term and with BatchNormalization. If it is nonlinear, then according to the number of projection layers $n$, we add $n-1$ `LinearLayers` with bias and BatchNormalizationRelu, and a `LinearLayer`. For the `SupervisedHead`, it is only a LinearLayer.

### ResNet_SimCLR
In the `resnet_simclr.py`, we implement the ResNet of the SimCLR. We know that according to the depth of the ResNet model, it can be devided into ResNet-34, ResNet-50, ResNet-101 and so on. They have similar structures, for example, all of the ResNet models can be devided into four groups, each group has a number of Residual blocks or Bottleneck blocks with the same number of filters. Also, in ResNet, before the Redisual blocks, it has an **input stem**. In the original paper, the input stem is consisted with a 7 \* 7 convolution layer with strides in 2. In the new paper (https://arxiv.org/abs/1812.01187), a new input stem was proposed. It is consisted with three 3 \* 3 convolution layers.

In `resnet`, we define the `block` and `layers` of different kinds of ResNet models. `block` is the kind of the ResNet blocks, and the `layers` is the number of ResNet blocks in each group. In class `ResNet`, we define the **input stem** and the **ResNet blocks group** according to the `resnet`. The **ResNet blocks group** is defined in `BlockGroup`.
