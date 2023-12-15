# ResNet_OCT_Disease

The dataset in kaggle (https://www.kaggle.com/datasets/paultimothymooney/kermany2018) is an OCT dataset organized into 3 folders (train, test, val) and contains subfolders for each image category (NORMAL,CNV,DME,DRUSEN). I learn a lot from this github page (https://github.com/anoopsanka/retinal_oct), and I write a naive version and upload it in here. It incorporate a reanet as a basis model and some convolution layers. 

After I learn more about self-supervised model and get better understanding I would upload a better version here.

## SimCLR
SimCLR is a simple framework for contrasive learning. Contrasive learning is unsupervised learning, which is used in classification task. The principales of the contrasive learning is that it calculated the distances between the input and positive reference and negative reference, and minimize the first distance, maximize the second distance. For the SimCLR, it thinks that the category of the images should not change with the color, observing angle and size. It uses augmentation methods to produce two versions of input, and use ResNet model to represent two versions of the images. Finally, use the cosine similarity to measure the distance between two versions. The model wants to minimize the distance between the same categories.

This model is simple and can achieve high accuracy. However, it needs large batch size.
