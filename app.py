app.py
import tkinter as tk
from tkinter import filedialog
import numpy as np
import librosa
import pickle
import sounddevice as sd
from scipy.io.wavfile import write


model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

def extract_features(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=16000, duration=3)

        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        mel = librosa.feature.melspectrogram(y=audio, sr=sr)

        features = np.hstack([
            np.mean(mfcc.T, axis=0),
            np.mean(chroma.T, axis=0),
            np.mean(mel.T, axis=0)
        ])
        return features
    except:
        return None

def decision(conf):
    if conf >= 0.85:
        return "High Confidence"
    elif conf >= 0.65:
        return "Needs Review"
    else:
        return "Uncertain"

def predict_file(file_path):
    features = extract_features(file_path)

    if features is None:
        status_label.config(text="Error processing file", fg="red")
        return

    features = scaler.transform([features])

    pred = model.predict(features)[0]
    prob = model.predict_proba(features)[0]
    conf = max(prob)

    label = "Human Voice" if pred == 0 else "Machine Voice"

    color = "green" if label == "Human Voice" else "blue"

    result_label.config(text=label, fg=color)
    conf_label.config(text=f"Confidence: {conf:.2f}")
    decision_label.config(text=f"Decision: {decision(conf)}")

    status_label.config(text="Done ✅", fg="green")

def upload_file():
    file_path = filedialog.askopenfilename(
        filetypes=[("Audio", "*.wav *.flac")]
    )

    if file_path:
        status_label.config(text="Processing...", fg="orange")
        app.update()
        predict_file(file_path)

def record_audio():
    fs = 16000
    duration = 3

    status_label.config(text="Recording...", fg="orange")
    app.update()

    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()

    write("recorded.wav", fs, recording)

    status_label.config(text="Processing recorded audio...", fg="orange")
    app.update()

    predict_file("recorded.wav")

app = tk.Tk()
app.title("Voice Classifier AI")
app.geometry("450x420")
app.configure(bg="#1e1e2f")

title = tk.Label(app, text="🎤 Voice Classification System",
                 font=("Arial", 16, "bold"),
                 bg="#1e1e2f", fg="white")
title.pack(pady=20)

# Upload button
tk.Button(app, text="Upload Audio",
          command=upload_file,
          font=("Arial", 12, "bold"),
          bg="#4CAF50", fg="white",
          width=20).pack(pady=10)

# Record button
tk.Button(app, text="🎤 Record Voice",
          command=record_audio,
          font=("Arial", 12, "bold"),
          bg="#FF5722", fg="white",
          width=20).pack(pady=10)

# Output labels
result_label = tk.Label(app, text="Prediction",
                        font=("Arial", 14, "bold"),
                        bg="#1e1e2f", fg="white")
result_label.pack(pady=10)

conf_label = tk.Label(app, text="Confidence",
                      font=("Arial", 12),
                      bg="#1e1e2f", fg="white")
conf_label.pack()

decision_label = tk.Label(app, text="Decision",
                          font=("Arial", 12),
                          bg="#1e1e2f", fg="white")
decision_label.pack()

status_label = tk.Label(app, text="Ready",
                        font=("Arial", 10),
                        bg="#1e1e2f", fg="yellow")
status_label.pack(pady=20)

app.mainloop()
