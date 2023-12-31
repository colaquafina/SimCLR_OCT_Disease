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
