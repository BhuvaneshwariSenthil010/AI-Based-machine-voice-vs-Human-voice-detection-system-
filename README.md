Human vs AI Voice Classification System

**Description**
 
This project is a Machine Learning-based offline desktop application that classifies audio input as either a Human Voice or an AI-Generated (Machine) Voice.
With the rapid growth of synthetic speech technologies, it has become important to distinguish between real and artificially generated voices. This system addresses that problem by analyzing audio signals and making intelligent predictions using trained ML models.
The application supports both audio file upload and live voice recording, making it practical for real-world usage.

**Methodology**

The system follows a structured pipeline:

1.	Audio Input

      o	Upload audio file (.wav / .flac) 
      o	Record live voice using microphone

2.	Feature Extraction

      o	MFCC (Mel-Frequency Cepstral Coefficients) 
      o	Chroma Features 
      o	Mel Spectrogram 

3.	Preprocessing
   
      o	Fixed duration (3 seconds) 
      o	Sample rate: 16 kHz 
      o	Feature normalization using StandardScaler

4.	Model Training
   
      o	Support Vector Machine (SVM with RBF kernel) 
      o	Random Forest (alternative model)
 
5.	Prediction Output

      o	Voice Classification (Human / AI) 
      o	Confidence Score 
      o	Decision Level 
________________________________________
**Decision Logic**

The system uses confidence-based classification:

•	≥ 0.85 →  High Confidence 

•	0.65 – 0.85 → Needs Review 

•	< 0.65 → Uncertain 

This ensures more reliable and interpretable predictions.
________________________________________
 **Features**
 
•	 Upload audio files 

•	 Record live voice 

•	 Machine Learning-based prediction 

•	 Confidence scoring system 

•	Simple and interactive GUI using Tkinter 

•	 Fully offline application (no internet required) 
________________________________________
**Technologies Used**
 
•	Python 
•	Tkinter (GUI) 
•	Librosa (Audio Processing) 
•	Scikit-learn (Machine Learning) 
•	NumPy 
________________________________________
📂 Project Workflow

Audio Input → Feature Extraction → Normalization → ML Model → Prediction → Decision Output
________________________________________
 How to Run
 
Step 1: Train the model

python train.py

 Step 2: Run the application
 
python app.py
________________________________________
**Sample Output**

Prediction: AI Generated Voice

Confidence: 0.91

Decision: High Confidence
________________________________________
**Applications**
 
•	Deepfake voice detection 

•	Voice authentication systems 

•	AI-generated content verification 

•	Security and fraud detection 
________________________________________
**Conclusion**
 
This project demonstrates how Machine Learning and audio signal processing can be combined to build an effective system for detecting AI-generated voices. It is designed to be simple, efficient, and fully offline, making it suitable for academic projects, hackathons, and real-world applications

# AI-Based-machine-voice-vs-Human-voice-detection-system-
