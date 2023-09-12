# Image Captioning with VGG16 and LSTM

This project focuses on creating an image captioning system that generates descriptive captions for images, with a special emphasis on images featuring dogs. The system utilizes the Flickr8k dataset, comprising 8,000 images each paired with five human-annotated captions.

## Dataset

The Flickr8k dataset contains a diverse range of images covering various objects, scenes, and activities, making it an ideal choice for training an image captioning model. 

## Model Components

### 1. VGG16 Pretrained Model
   - The VGG16 model, pre-trained on the ImageNet dataset, is employed for extracting meaningful visual features from the images.

### 2. LSTM Model
   - An LSTM (Long Short-Term Memory) neural network, known for capturing sequential information and context, is used for generating captions based on the visual features extracted by the VGG16 model.

## Problem Steps

### Data Preprocessing
   - Load and organize the Flickr8k dataset, including images and their corresponding captions.
   - Split the dataset into training and test sets for model evaluation.
   - Preprocess the images by resizing them to a fixed size and normalizing pixel values.
   - Tokenize the captions and build a vocabulary to convert words into numerical indices.

### VGG16 Feature Extraction
   - Employ the pre-trained VGG16 model to extract visual features from the input images.
   - Remove fully connected layers from the VGG16 model to obtain the feature vector.

### LSTM Model for Caption Generation
   - Design an LSTM model that takes the VGG16 feature vector as input and generates captions.
   - Implement an embedding layer to convert word indices to dense vectors.
   - Utilize the LSTM layer to capture sequential information from the embedded captions.
   - Add a dense layer with softmax activation to predict the next word in the caption.
 ![model](https://github.com/itsmethahseer/Image-Captioning/assets/120078997/48785e3b-1ae3-45de-8c21-4b4570387908)
### Training
   - Train the LSTM model on the training set using the VGG16 feature vectors and corresponding captions.
   - Utilize categorical cross-entropy loss to measure the discrepancy between predicted and ground truth captions.
   - Use the Adam optimizer to update model weights during training.

### Caption Generation
   - After training, employ the trained LSTM model along with the VGG16 feature vectors to generate captions for new images.

### Evaluation
   - Assess the performance of the image captioning system using standard evaluation metrics like BLEU.
   - Compare the generated captions against the ground truth captions to measure system accuracy and quality.

## Conclusion

By combining the power of the pretrained VGG16 model and LSTM, this image captioning system provides a robust tool for automatically generating accurate and descriptive captions for a wide range of images, effectively bridging the gap between vision and language tasks.

## Reference

Image caption description by Andrej Karpathy.

---

