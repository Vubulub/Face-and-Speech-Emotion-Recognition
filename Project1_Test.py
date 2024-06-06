# import necessary libraries
import pickle
import pyaudio
import numpy as np
import librosa
import argparse
import cv2
import os
import random
import threading
import speech_recognition as sr
from keras.models import model_from_json
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# suppress TensorFlow warning messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# load the speech emotion detection model
json_file = open('models/speech_emotion_detection_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("models/speech_emotion_detection_model.h5")

# load the scaler
with open('scaler.pickle', 'rb') as f:
    scaler = pickle.load(f)

# define emotion labels and a dictionary
emotion_labels = {0: 'neutral', 1: 'calm', 2: 'happy', 3: 'sad', 4: 'angry', 5: 'fearful', 6: 'disgust', 7: 'surprised'}
emotion_dict = {0: "angry", 1: "disgusted", 2: "fearful", 3: "happy", 4: "neutral", 5: "sad", 6: "surprised"}

# function to extract audio features
def extract_features(data, sample_rate):
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    return np.concatenate((zcr, chroma_stft, mfcc, rms, mel))

# function to predict emotion
def predict_emotion(features):
    scaled_features = scaler.transform(features.reshape(1, -1))
    reshaped_features = np.expand_dims(scaled_features, axis=2)
    predictions = loaded_model.predict(reshaped_features)
    emotion_index = np.argmax(predictions)
    emotion_label = emotion_labels[emotion_index]
    return emotion_label

# audio settings
FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 22050
CHUNK = 1024
SECONDS = 0.1
THRESHOLD = 0.0001

audio = pyaudio.PyAudio()

# open audio stream
stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=int(RATE * SECONDS))

print("Listening for speech...")

emotion = ""

# function to process audio in a separate thread
def process_audio():
    global emotion
    while True:
        data = stream.read(int(RATE * SECONDS))
        audio_data = np.frombuffer(data, dtype=np.float32)
        energy = np.sum(audio_data ** 2) / len(audio_data)
        if energy > THRESHOLD:
            features = extract_features(audio_data, RATE)
            emotion = predict_emotion(features)

audio_thread = threading.Thread(target=process_audio)
audio_thread.daemon = True
audio_thread.start()

# argument parser
ap = argparse.ArgumentParser()
ap.add_argument("--mode", help="display")
mode = ap.parse_args().mode

# directories and parameters for image data generator
train_dir = 'train'
val_dir = 'test'

num_train = 28709
num_val = 7178
batch_size = 64
num_epoch = 50

# image data generators
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

# train generator
train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(48, 48),
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode='categorical')

# validation generator
validation_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(48, 48),
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode='categorical')

# define the CNN model
model = Sequential()

# add convolutional layers to the model
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# flatten the layers and add dense layers
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

# load the model weights if in test mode
if mode == "test":
    model.load_weights('models/face_emotion_model.h5')

    # initialize the webcam
    cv2.ocl.setUseOpenCL(False)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Couldn't open webcam")
        exit()

    # initialize speech recognition
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    speech_text = ""

    # function to recognize speech
    def recognize_speech():
        global speech_text
        while True:
            try:
                with mic as source:
                    audio = recognizer.listen(source, timeout=1, phrase_time_limit=5)
                    speech_text = recognizer.recognize_google(audio)
            except sr.WaitTimeoutError:
                pass
            except sr.UnknownValueError:
                speech_text = "Could not understand audio"
            except sr.RequestError as e:
                speech_text = f"Error: {e}"

    # start speech recognition thread
    speech_thread = threading.Thread(target=recognize_speech)
    speech_thread.daemon = True
    speech_thread.start()

    speech_emotions = list(emotion_labels.values())
    face_emotions = list(emotion_dict.values())
    speech_target = random.choice(speech_emotions)
    face_target = random.choice(face_emotions)
    speech_score = 0
    face_score = 0

    # main loop for webcam processing
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        face_emotion_detected = None

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
            prediction = model.predict(cropped_img)
            maxindex = int(np.argmax(prediction))
            accuracy = np.max(prediction) * 100
            text = f"{emotion_dict[maxindex]} {accuracy:.2f}%"
            cv2.putText(frame, text, (x + 20, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            face_emotion_detected = emotion_dict[maxindex]

        if emotion == speech_target:
            speech_target = random.choice(speech_emotions)
            speech_score += 1

        if face_emotion_detected == face_target:
            face_target = random.choice(face_emotions)
            face_score += 1

        cv2.putText(frame, f"Speech Target: {speech_target}", (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2,
                    cv2.LINE_AA)
        cv2.putText(frame, f"Face Target: {face_target}", (0, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2,
                    cv2.LINE_AA)
        cv2.putText(frame, f"Speech Score: {speech_score}", (470, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2,
                    cv2.LINE_AA)
        cv2.putText(frame, f"Face Score: {face_score}", (470, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2,
                    cv2.LINE_AA)

        cv2.putText(frame, f"Speech Emotion: {emotion}", (10, frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, speech_text, (10, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('Video', cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_CUBIC))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
