# Project overview
- The project uses transfer learning on the Inception-v3 model to learn how to use the pre-trained model and gain access to knowledge about transfer learning and the Inception-v3 architecture. In this project, I tested on a set of human horses. It's surprising that the battery is up to 99.39%. What an unexpected result.

## 1.	Theory
### 1.1. Inception V1
- When multiple deep layers of convolutions were used in a model it resulted in the overfitting of the data. To avoid this from happening the inception V1 model uses the idea of using multiple filters of different sizes on the same level. Thus in the inception models instead of having deep layers, we have parallel layers thus making our model wider rather than making it deeper.
- The Inception model is made up of multiple Inception modules.
- The basic module of the Inception V1 model is made up of four parallel layers.
	- 1×1 convolution
	- 3×3 convolution
	- 5×5 convolution
	- 3×3 max pooling
- Convolution : The process of transforming an image by applying a kernel over each pixel and its local neighbors across the entire image.
- Pooling : Pooling is the process used to reduce the dimensions of the feature map. There are different types of pooling but the most common ones are max pooling and average pooling.
### 1.2. Inception V3 Model Architecture
- Inception v3 is an image recognition model that has been shown to attain greater than 78.1% accuracy on the ImageNet dataset. The model is the culmination of many ideas developed by multiple researchers over the years.[Reference](https://cloud.google.com/tpu/docs/inception-v3-advanced)

- The inception v3 model was released in the year 2015, it has a total of 42 layers and a lower error rate than its predecessors. Let's look at what are the different optimizations that make the inception V3 model better.

- The major modifications done on the Inception V3 model are
	- Factorization into Smaller Convolutions
	- Spatial Factorization into Asymmetric Convolutions
	- Utility of Auxiliary Classifiers
	- Efficient Grid Size Reduction
- After performing all the optimizations the final Inception V3 model looks like this

![inceptionv3onc--oview_vjAbOfw](https://github.com/FPT-ThaiTuan/Transfer-Learning-Use-Inception-v3-For-Image-Classification/assets/105273233/39259061-da24-4a7b-818f-d2d7f7c165c0)

- In total, the inception V3 model is made up of 42 layers which is a bit higher than the previous inception V1 and V2 models. But the efficiency of this model is really impressive. We will get to it in a bit, but before it let's just see in detail what are the components the Inception V3 model is made of.

![屏幕截图 2024-03-30 114642](https://github.com/FPT-ThaiTuan/Transfer-Learning-Use-Inception-v3-For-Image-Classification/assets/105273233/0abe7578-e07c-40fa-94d7-0dc72e21a043)

### 1.3. Transfer learing
#### 1.3.1. Theory
- Transfer learning is a machine learning method where a model already developed for a task is reused in another task. Transfer learning is a popular approach in deep learning, as it enables the training of deep neural networks with less data compared to having to create a model from scratch.
- Typically, training a model takes a large amount of compute resources and time. Using a pre-trained model as a starting point helps cut down on both. [Reference](https://www.techtarget.com/searchcio/definition/transfer-learning)

![how_transfer_learning_works-f](https://github.com/FPT-ThaiTuan/Transfer-Learning-Use-Inception-v3-For-Image-Classification/assets/105273233/0946f9c8-e89a-40d1-9b19-39c1a0fb70e5)

#### 1.3.2. The role of transfer learning
- Knowledge transfer
- Improve accuracy and save training costs
- Efficient with small data
#### 1.3.3. Warm up
- Warm-up is a crucial step in training to expedite convergence. During this phase, the CNN layers are frozen, preserving their learned coefficients, while only the last Fully Connected Layers are retrained. The aim of warm-up is to maintain the valuable high-level features acquired from the pretrained-model, which are advantageous due to being trained on a larger and more accurate dataset. This approach yields higher accuracy compared to random coefficient initialization.
#### 1.3.4. Fine tuning model
- The primary objective of warming up the model is to expedite convergence towards the global optimum during training. Once the model achieves satisfactory performance on the fully connected layers, further improvements in accuracy become challenging. At this point, unfreezing the layers of the base network and training the entire model, including the pretrained layers, is necessary. This process is known as fine-tuning.
#### 1.3.5. Experience in transfer learning
- Transfer learning is a valuable technique that leverages pretrained models to enhance the performance of a target model, particularly when dealing with limited data. Here's a summary of considerations and guidelines:

- Data Size and Retraining Strategy:

	- Small Data: Retrain only the last fully connected layers to avoid losing features learned from the pretrained model, which could lead to inaccurate predictions.
	- Large and Domain-like Data: Retrain all layers, but consider using a warm-up step and fine-tuning to expedite the training process.
	- Large Data and Different Domains: Retrain the entire model from scratch, as features learned from the pretrained model may not be beneficial for data from distinct domains.
 
- Applicability of Transfer Learning:

	- Transfer learning is effective when both models (pretrained-model A and model to be trained B) belong to the same domain. Features learned from pretrained-model A are likely to be useful for model B's classification task.
	- Transfer learning is most effective when the training data for pretrained-model A is larger than that of model B. Features learned from a larger dataset tend to generalize better.
	- Pretrained-model A should exhibit good qualities, as only models with robust feature representations can contribute effectively to the performance of model B.

- In summary, transfer learning should be applied judiciously based on the compatibility of domains, the size of the training data, and the quality of the pretrained model. It is most effective when these factors align appropriately with the target task and dataset. [Reference](https://phamdinhkhanh.github.io/2020/04/15/TransferLearning.html)

## 2. Practice projects
### 2.1. Dataset
### 2.1.1. Collect dataset
- The human - horse dataset is collected from [Dataset](https://laurencemoroney.com/datasets.html#google_vignette)
### 2.1.2. Create training and validation data for the model
- Create a data set with the process of reading and writing files
### 2.1.3. Data Augmentation
- ImageDataGenerator is a tool in Keras that enhances training data by applying different transformations to images. Its properties such as rescale, rotation_range, zoom_range, shear_range, width_shift_range, height_shift_range, horizontal_flip and vertical_flip help create diversity in training data, thereby improving the generalization and stability of deep learning models.
### 2.2. Buil model
#### 2.2.1. Download the pre-trained model
- Here I use the Inception V3 model
#### 2.2.2. Warm up
- Do Warm up from the last layer named 'mixed7'
- Freeze the layers in front of it and change the output
#### 2.2.3. Complete model and test
- Train the model with the augmented dataset with output using the sigmoid function
- Perform testing on datasets to enhance validation
#### 2.2.4. Show result and save model
- Model fitting

![屏幕截图 2024-03-30 134026](https://github.com/FPT-ThaiTuan/Transfer-Learning-Use-Inception-v3-For-Image-Classification/assets/105273233/47b468a9-11ae-4a7c-b50a-646cfd838381)

- Plotting training loss and validation

![屏幕截图 2024-03-30 134421](https://github.com/FPT-ThaiTuan/Transfer-Learning-Use-Inception-v3-For-Image-Classification/assets/105273233/6eae6abc-dd44-4973-b7b6-3d742ad200a5)

- Plot the training and validation accuracy

![屏幕截图 2024-03-30 134453](https://github.com/FPT-ThaiTuan/Transfer-Learning-Use-Inception-v3-For-Image-Classification/assets/105273233/dcf7ab38-c493-4dd9-b20c-c1ecdbdef144)

## 3. Top pre-trained model
- Pre-trained models in NLP are BERT, GPT-3, XLNet, RoBERTa, and DistilBERT. BERT is famous for its bidirectional .
- The models are pre-trained in image processing  VGG, Xception, ResNet, Inception


**Hope this article can help you.**

**If you have any questions please contact me for help!**

**Gmail: tuanddt.ai.work@gmail.com**

***Thanks everyone!***
