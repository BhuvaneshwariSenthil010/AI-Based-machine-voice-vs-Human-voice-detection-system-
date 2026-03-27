train.py:
import os
import numpy as np
import librosa
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score

def extract_features(file_path):
    try:
        # Load only 3 sec for consistency
        audio, sr = librosa.load(file_path, sr=16000, duration=3)

        # MFCC
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)

        # Chroma
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)

        # Mel Spectrogram
        mel = librosa.feature.melspectrogram(y=audio, sr=sr)

        # Combine features
        features = np.hstack([
            np.mean(mfcc.T, axis=0),
            np.mean(chroma.T, axis=0),
            np.mean(mel.T, axis=0)
        ])

        return features

    except Exception as e:
        print("Error:", file_path)
        return None


human_path = "dataset/human"
machine_path = "dataset/machine"

X = []
y = []


for file in os.listdir(human_path)[:400]:
    f = extract_features(os.path.join(human_path, file))
    if f is not None:
        X.append(f)
        y.append(0)


for file in os.listdir(machine_path)[:400]:
    f = extract_features(os.path.join(machine_path, file))
    if f is not None:
        X.append(f)
        y.append(1)

# Convert to numpy
X = np.array(X)
y = np.array(y)

print("Dataset shape:", X.shape)


X, y = shuffle(X, y, random_state=42)


scaler = StandardScaler()
X = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = SVC(kernel='rbf', probability=True)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

print("Model saved successfully!")