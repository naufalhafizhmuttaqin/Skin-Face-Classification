### Skin Type Classification
This dataset contains images of faces with different skin types, such as:
- Normal 😊
- Oily 💦
- Dry 🌵
- Acne 😓

Kaggle Dataset = https://www.kaggle.com/datasets/muttaqin1113/face-skin-type

I employed two Convolutional Neural Network (CNN) models—MobileNetV2 and InceptionV3—to classify skin types and provide skincare recommendations based on the classification results. The methodology includes data preprocessing, model training, and evaluation.
The dataset is split into training and test sets. The training set is further divided into training and validation sets to fine-tune the model. This process continues iteratively until optimal model performance is achieved. Finally, the test set is used to evaluate the model’s generalization to new, unseen data.

### Model Architecture
In this project, a pre-trained Convolutional Neural Network (CNN)—specifically InceptionV3—is used as the backbone model for feature extraction. The original top layers of the model are removed (include_top=False), and custom classification layers are added on top. The model is trained using transfer learning, followed by fine-tuning to optimize performance on the skin type classification task.

The full architecture consists of the following components:
- **Input Layer** 

Accepts RGB images resized to 299x299 pixels as required by the InceptionV3 model and images resized to 224x224 pixels as required by the MobileNetV2

- **Pre-trained Convolutional Base**

Utilizes the InceptionV3 model with pre-trained weights from ImageNet for feature extraction. All layers are initially frozen to prevent updating during the early training phase.

- **Global Average Pooling Layer**

Reduces the spatial dimensions of the feature maps, making the model more efficient and less prone to overfitting compared to flattening.

- **Batch Normalization**/n
Applied to stabilize and accelerate training by normalizing the activations.

- **Fully Connected Layer**

A dense layer with 256 neurons and ReLU activation to learn complex patterns from the extracted features.

- **Dropout Layer**

Applies dropout with a rate of 0.5 to prevent overfitting by randomly disabling neurons during training.

- **Output Layer**

A dense layer with 4 output units and softmax activation, corresponding to the four skin type classes:
- Normal
- Oily
- Dry
- Acne

This architecture enables efficient transfer learning and is further improved through fine-tuning after the initial training phase, where all layers of the pre-trained base are unfrozen and trained with a lower learning rate.


### Requirements
- **Python**
- **Keras** 
- **TensorFlow** 
- **NumPy** 
- **Matplotlib** 
- **sckit-learn**
- **pandas**
- **seaborn**
