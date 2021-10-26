# Convolutional-Neural-Network-101-Novice

## Architechture
![1_uulvWMFJMidBfbH9tMVNTw@2x](https://user-images.githubusercontent.com/86042628/138884826-bc3d312e-f3d2-4fcb-af9c-fef09e8f0458.png)


## Feature Extraction
1. **Filter** an image for a particular feature (convolution)
2. **Detect** that feature within the filtered image (ReLU)
3. **Condense** the image to enhance the features (maximum pooling)
 
![IYO9lqp (2) (2)](https://user-images.githubusercontent.com/86042628/138882728-61ae5855-3172-4aaa-a3e4-16738baba747.jpg)

Typically, the Convolution network will perform *several extractions in parallel on a single image*. In modern convnets, it's not uncommon for the final layer in the base to be producing over 1000 unique visual features.
With each process image size will keep on decreaseing (**Image size decreases with each layer**)

![1_AkxFKERBmX1hZS4tKQSU4w](https://user-images.githubusercontent.com/86042628/138883719-2765be9f-24b1-4339-a835-5efbc52c2a18.png)
![1_md0LLXeexbcA8AnUC7It8w](https://user-images.githubusercontent.com/86042628/138883911-a4096e43-7021-4d3d-8322-48b5d056ed7d.png)

original image ![1_nmN4V6FH8G8lGunVrTpVlw](https://user-images.githubusercontent.com/86042628/138884326-99d5c562-a038-4295-8d75-9eccca12eade.jpg)


### Feature maps
The activations in the network we call feature maps. They are what result when we apply a filter to an image; they contain the visual features the kernel extracts. Here are a few kernels pictured with feature maps they produced.
![1_VVvdh-BUKFh2pwDD0kPeRA@2x](https://user-images.githubusercontent.com/86042628/138885066-33844db5-a3b4-4486-9c8c-9ab562f622aa.gif)
![1_oxOsZPfZFxgGZw2ycQnenw@2x (1)](https://user-images.githubusercontent.com/86042628/138885487-1696f93d-6725-4d90-8cce-8fcfd931c9c7.png)

### Stride and Padding
* Stride specifies how much we **move the convolution filter at each step**. By default the value is 1
* We use **padding to surround the input with zeros.**
![1_L4T6IXRalWoseBncjRr4wQ@2x](https://user-images.githubusercontent.com/86042628/138886013-55fb9399-5bd2-4744-9a11-5a8f955b8e70.gif) ![1_4wZt9G7W7CchZO-5rVxl5g@2x](https://user-images.githubusercontent.com/86042628/138886031-bfd0184f-25c1-48a4-b1da-5c1b8a511d1b.gif)
![1_W2D564Gkad9lj3_6t9I2PA@2x](https://user-images.githubusercontent.com/86042628/138886047-866bc43d-3598-43f1-be8e-515ad3078a01.gif)

### Sliding Window
Stirde and padding are also called **Sliding Windows** 
It's def as per [kaggle]('https://www.kaggle.com/ryanholbrook/the-sliding-window')
The strides parameter says how far the window should move at each step, and the padding parameter describes how we handle the pixels at the edges of the input.
padding with eg
For a gray scale (n x n) image and (f x f) filter/kernel, the dimensions of the image resulting from a convolution operation is (n – f + 1) x (n – f + 1). 
p = number of layers of zeros added to the border of the image, then our (n x n) image becomes (n + 2p) x (n + 2p) image after padding. So, applying convolution-operation (with (f x f) filter) outputs (n + 2p – f + 1) x (n + 2p – f + 1) images. 
**Valid Padding:** It implies no padding at all. The input image is left in its valid/unaltered shape. 
So, [(n x n) image] * [(f x f) filter] —> [(n – f + 1) x (n – f + 1) image]  {where * represents a convolution operation}

**Same Padding:** In this case, we add ‘p’ padding layers such that the output image has the same dimensions as the input image. 
So, 
[(n + 2p) x (n + 2p) image] * [(f x f) filter] —> [(n x n) image]
which gives `p = (f – 1) / 2` (because n + 2p – f + 1 = n). 
So, if we use a (the 3 x 3) filter the 1 layer of zeros must be added to the borders for the same padding. Similarly, if (5 x 5) filter is used 2 layers of zeros must be appended to the border of the image.

### Pool
**This is a condensing step to the model**
This is used to **reduce the dimensionality**. This enables us to `reduce the number of parameters, which both shortens the training time and combats overfitting`. Pooling layers **downsample each feature map independently**, reducing the height and width, keeping the depth intact.

A MaxPool2D layer is much like a Conv2D layer, except that it uses a simple maximum function instead of a kernel, with the pool_size parameter analogous to kernel_size. A MaxPool2D layer doesn't have any trainable weights like a convolutional layer does in its kernel.  
**Max pooling takes a patch of activations in the original feature map and replaces them with the maximum activation in that patch.**
![pool](https://user-images.githubusercontent.com/86042628/138878160-aff2fa2a-6529-4c5d-ba4b-93770beeae1c.png)

#### Types of Pooling
1. Average Pooling : Average pooling gives the average of features present in a patch. 
<img width="695" alt="Screenshot-2019-07-21-at-3 05 56-AM" src="https://user-images.githubusercontent.com/86042628/138880820-d8347a35-42f2-483b-9101-90f2390ba19f.png">
2. Max Pooling : pooling operation that selects the maximum element from the region of the feature map covered by the filter.
<img width="695" alt="Screenshot-2019-07-21-at-2 57 13-AM" src="https://user-images.githubusercontent.com/86042628/138880923-a1332695-eb4d-4cb2-a828-05ff1737ca10.png">



### Translation Invariance
When MaxPool2D removes some of these pixels, `it removes some of the positional information in the feature map`.   
This gives a convnet a **property** called translation invariance. This means that a `convnet with maximum pooling will tend not to distinguish features by their location in the image`.  
("Translation" is the mathematical word for changing the position of something without rotating it or changing its shape or size.)

Watch what happens when we `repeatedly apply maximum pooling` to the following feature map.

![97j8WA1](https://user-images.githubusercontent.com/86042628/138880325-9485199d-fe97-45e0-9202-01a0444ee6ea.png)

"In Plane English": Invariance means that you can recognize an object as an object, even when its appearance varies in some way. This is generally a good thing, because it preserves the object's identity, category, (etc) across changes in the specifics of the visual input, like relative positions of the viewer/camera and the object.

The image below contains many views of the same statue. You (and well-trained neural networks) can recognize that the same object appears in every picture, even though the actual pixel values are quite different.

![translation Invariance](https://user-images.githubusercontent.com/86042628/138879995-fc87183e-649a-464a-b73e-9bb249147c06.png)


**Another way to get invariance is with Data Augmentation**
### Data Augmentation
Data augmentation is a way to **generate more training data from our current set**. It `enriches or “augments” the training data by generating new examples` via **random transformation of existing ones**. This way we artificially **boost** the **size of the training set**, **reducing overfitting**. So data augmentation can also be considered as a **`regularization technique`.**
original 
![1_H3E66N_7umdrobakY-wG4A@2x](https://user-images.githubusercontent.com/86042628/138894316-4e6ea8b7-fb09-4614-9efa-cdd46dd1be03.png)

After Augmenting, if we can infer it as a human, so the model should be able to learn that as well.
![1_S-AIjI0q1Fj9ni20NBkIYA@2x](https://user-images.githubusercontent.com/86042628/138894448-51c44b5c-46f7-4e58-9e94-f27bddbd8e71.png)
![UaOm0ms](https://user-images.githubusercontent.com/86042628/138894612-e008e9be-7da6-43a4-b794-0a163aa53ed0.png)


## Hyperparameters
**layers.Conv2D(filters=32, kernel_size=5, activation="relu", padding='same'),**
1. **Filter size:** we typically use 3x3 filters, but 5x5 or 7x7 are also used depending on the application. There are also 1x1 filters which we will explore in another article, at first sight it might look strange but they have interesting applications. Remember that these filters are 3D and have a depth dimension as well, but since the depth of a filter at a given layer is equal to the depth of its input, we omit that.
2. **Filter count:** this is the most variable parameter, it’s a power of two anywhere between 32 and 1024. Using more filters results in a more powerful model, but we risk overfitting due to increased parameter count. Usually we start with a small number of filters at the initial layers, and progressively increase the count as we go deeper into the network.
3. **Stride:** we keep it at the default value 1.
4. **Padding:** we usually use padding.

## Dropout
*Fun Fact:* Even the state-of-the-art models which have 95% accuracy get a 2% accuracy boost just by adding dropout, which is a fairly substantial gain at that level.
Dropout is used to **prevent overfitting**
During training time, at each iteration, a neuron is temporarily “dropped” or disabled with probability p. This means all the inputs and outputs to this neuron will be disabled at the current iteration. The dropped-out neurons are resampled with probability p at every training step, so a dropped out neuron at one step can be active at the next one. The hyperparameter p is called the dropout-rate and it’s typically a number around 0.5, corresponding to 50% of the neurons being dropped out.
It’s surprising that dropout works at all. We are disabling neurons on purpose and the network actually performs better. The reason is that dropout prevents the network to be too dependent on a small number of neurons, and forces every neuron to be able to operate independently. This might sound familiar from constraining the code size of the autoencoder in Part 3, in order to learn more intelligent representations.
Let’s visualize dropout, it will be much easier to understand.


## This repository contains 
1. Fashion MNIST 
2. CIFAR
