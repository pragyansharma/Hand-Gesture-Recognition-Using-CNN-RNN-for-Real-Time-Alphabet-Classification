# Hand-Gesture-Recognition-Using-CNN-RNN-for-Real-Time-Alphabet-Classificationt
This repository presents a Hand Gesture Recognition System designed to classify static and sequential hand gestures into the 26 English alphabet letters (A-Z) using Convolutional Neural Networks (CNN) and Recurrent Neural Networks (RNN).
Hand gesture recognition is of great significance in artificial intelligence that bounds human computing interface and computer vision. This project put in place a well designed hand gesture recognition system through the use of CNN and RNN. Static and sequential gestures are to be classified under the twenty-six alphabets of the English language as envisaged by the model. Data collection, data preprocessing, and the model training, all were designed and implemented with high attention to the quality. The final system obtained 100 percent accuracy in the desired validation set and it is a real time program. Further work will introduce multilingual gestures and make adjustments for better applicability in a wide variety of settings.

**Introduction and Background**
Hand gesture recognition is an emerging field of artificial intelligence, particularly within computer vision. It facilitates seamless human-computer interaction, allowing intuitive interfaces without traditional input devices. Our project focuses on recognizing gestures representing the English alphabet, leveraging deep learning techniques to classify both static and sequential gestures.
![image](https://github.com/user-attachments/assets/7a66808a-1b0c-4b09-bad4-8ba0bed7db5f)

**The problem you tried to solve**
The challenge lies in creating an accurate and efficient hand gesture recognition system that can classify gestures representing the English alphabet (A-Z) in real-time. This involves overcoming challenges like variations in hand sizes, gestures, lighting, and camera angles.
**1.2 Results from the literature**
Several studies have explored gesture recognition using CNNs for static images and RNNs for temporal dependencies.
•	Zhang et al. (2021) explored hand gesture recognition using CNN-LSTM models and demonstrated over 90% accuracy[1].
•	Khan et al. (2020) focused on data augmentation techniques to improve the robustness of CNN-based models in gesture recognition[2].
•	Our project builds on these approaches but extends them with a unique dataset and real-time deployment, improving accuracy and efficiency for alphabet recognition.

**1.3 What tools and programs are already available for the problem, or for closely related ones?**
For model implementation we utilized TensorFlow and Keras for the model implementation part while for data collection and preprocessing we employed OpenCV and Matplotlib for the visualization of our results. During the initial data collection involving real-time gesture capture, tools such as cvzone was useful in hand detection. All these tools were very much applicable that built very strong system.

**2 Overview of the architecture**
For hand gesture recognition using deep learning for Our program suite is a modular system for the following reasons: Architecture of the solution is divided into several components that rely on each other to perform efficiently and effectively.
**2.1 Finished work: CNN-RNN Model**


![image](https://github.com/user-attachments/assets/2a1982d7-d567-4da1-8a4f-066ed51e764b)

•	**TimeDistributed CNN Layers:**
o	Extract spatial features from each frame of the input sequence.
o	Layers include convolutional operations followed by max pooling for dimensionality reduction.
•	**LSTM Layer:**
o	Captures temporal dependencies across the input sequence.
o	Processes the output of the TimeDistributed CNN layers.
•**	Dense Layers:**
o	Fully connected layers for classification, with a final layer containing 26 output nodes for gesture classification (A-Z).
The model contains 42,572,634 trainable parameters, making it highly capable of learning complex spatial and temporal patterns.
**2.2 Work in progress: Modules designed but not implemented**
Integration of advanced augmentation techniques to improve robustness.
Expansion to multi-language gesture recognition
**2.3 Future work: Modules a future continuation may have**
Fine-tuning with domain adaptation for diverse lighting and environmental conditions.

**3 Data Collection**
We collected a dataset of over 6,000 hand gesture images representing the English alphabet (A-Z) using a webcam and OpenCV. Images were processed into sequences of 5 frames, resized to 150x150 pixels, and normalized for training. Variations in gestures, lighting, and angles were ensured to improve model robustness.
 ![image](https://github.com/user-attachments/assets/87cbc1ff-747e-4993-bbd8-2f88e7968a0b)

**4 Your methods and implementation **
The system uses a CNN for spatial feature extraction and an LSTM for temporal sequence modeling:
•	Preprocessing: Images were cropped using hand detection and resized to 150x150. Data was normalized to [0, 1].
•	Model Architecture:
o	CNN: TimeDistributed layers extract features from individual frames.
o	RNN: LSTM layers capture temporal patterns in sequences.
•	Training: The model was trained on 80% of the dataset, with 20% reserved for validation.

5 Results and Evaluation

![image](https://github.com/user-attachments/assets/ec7a1ce2-338f-4b74-a81e-7bc09681c59a)


![image](https://github.com/user-attachments/assets/e822cdee-cfce-4a51-838f-f9867e58f58a)

**Figure 1: Accuracy over Epochs**
•	Training Accuracy: The training accuracy rapidly improves, achieving 100% accuracy by the 5th epoch.
•	Validation Accuracy: The validation accuracy closely matches the training accuracy, achieving 100% by the 5th epoch, confirming excellent model performance on the validation set.

**Figure 2: Loss over Epochs**
•	Training Loss: The training loss steadily decreases, reaching near-zero values by the 5th epoch, indicating the model fits the training data well.
•	Validation Loss: The validation loss closely follows the training loss, showing no signs of overfitting. This suggests the model generalizes well to unseen data.


**6 Discussion and Conclusions**
Our project successfully implemented a robust hand gesture recognition system. The model demonstrated excellent performance on the validation set and in real-time testing. Future improvements include expanding the dataset to support multilingual gestures, optimizing for mobile deployment, and addressing complex lighting scenarios.

**Results:**
![image](https://github.com/user-attachments/assets/af86ff1f-f1aa-43cc-9a99-481b42e1b670)


**How to Use**
Prerequisites
**1 Create a Data Folder:**

Inside the project directory, create a folder named Data.
Organize subfolders for each gesture class (e.g., Data/A, Data/B, ..., Data/Z).
**Capture Gesture Data:**

Use the provided data_collection.py script to capture hand gestures for each class.
Setup Instructions

**Clone the Repository:**
git clone https://github.com/your-username/hand-gesture-recognition.git
cd hand-gesture-recognition

**Install Dependencies:**
pip install -r requirements.txt

**Collect Gesture Data**
python data_collection.py

**Train the Model :**
python train_model.py

Run Real-Time Gesture Recognition:
python real_time_gesture_recognition.py

Installing All Dependencies
