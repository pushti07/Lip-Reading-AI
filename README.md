
# Lip-Reading AI Using TensorFlow

This project implements a deep learning-based lip-reading AI system using TensorFlow. The goal of this system is to predict speech (text) based on video sequences of lip movements. The model is trained using 3D Convolutional Neural Networks (Conv3D) and Bidirectional LSTMs to capture both spatial and temporal information from video frames.

## Table of Contents
- Project Overview
- Dataset
- Model Architecture
- Installation
- Usage
- Results
- Contributing
- License

## Project Overview
Lip-reading AI translates visual information from lip movements into text, which can be useful in various applications, such as:
- Helping people with hearing disabilities.
- Enhancing speech recognition systems in noisy environments.
- Enabling more intuitive and human-like interaction with AI systems.

This project takes video data, processes it frame-by-frame, and uses machine learning models to predict the text being spoken in the video.

## Dataset
The project works with a custom dataset of videos and corresponding text alignments. Each video contains footage of a person speaking, and the alignment files contain the transcriptions of what is being said in each video.

The dataset used in this project contains:
- **Video Files**: .mpg format
- **Text Alignments**: Corresponding .align files that contain the transcriptions.

## Model Architecture
The architecture for this project consists of:
- **3D Convolutional Layers**: Used to process spatial information across multiple frames of the video.
- **Bidirectional LSTMs**: To capture the temporal dependencies and generate sequential text predictions.
- **TimeDistributed Flattening**: To feed 3D data into LSTM layers.
- **Dense Layer with Softmax Activation**: For character classification.

## Installation
### Prerequisites
- Python 3.x
- TensorFlow
- OpenCV
- ImageIO
- Gdown (for downloading datasets)

### Steps
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/lip-reading-ai.git
   cd lip-reading-ai
   ```

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the dataset (if available):
   ```bash
   gdown https://drive.google.com/uc?id=1YlvpDLix3S-U8fd-gqRwPcWXAXm8JwjL
   unzip data.zip
   ```

## Usage
### Training
You can start the training process by running:
```bash
python train.py
```

This will load the video data, preprocess it, and train the model on the dataset.

### Prediction
Once the model is trained, you can make predictions on new video data using:
```bash
python predict.py --video path_to_video.mpg
```

### Model Structure
To modify or experiment with the model structure, edit the `model.py` file.

## Results
After training, the model will output the predicted text for the video. You can further evaluate the model by computing accuracy or loss metrics during the training process.

### Sample Output
For a given input video, the model predicts the sequence of characters (speech) as follows:
```
Input Video: sample_video.mpg
Predicted Text: "hello world"
```

## Contributing
If you'd like to contribute to the project, feel free to submit a pull request or open an issue. Contributions can include improving model accuracy, adding features, or fixing bugs.

