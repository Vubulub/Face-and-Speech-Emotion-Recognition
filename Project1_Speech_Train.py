import os
import numpy as np
import librosa
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from keras.callbacks import ReduceLROnPlateau
import pickle

# define a function to extract audio features
def extract_features(data, sample_rate):
    result = np.array([])  # feature array
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)  # zero crossing rate
    result = np.hstack((result, zcr))  # stacking features
    stft = np.abs(librosa.stft(data))  # short-time Fourier transform
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)  # chroma feature
    result = np.hstack((result, chroma_stft))  # stacking features
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)  # Mel-frequency cepstral coefficients
    result = np.hstack((result, mfcc))  # stacking features
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)  # root mean square energy
    result = np.hstack((result, rms))  # stacking features
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)  # Mel spectrogram
    result = np.hstack((result, mel))  # stacking features
    return result

# define a function to get features from audio files in a folder
def get_features(folder_path, sample_rate):
    x, y = [], []  # initialize feature and label arrays
    for subdir, dirs, files in os.walk(folder_path):  # iterate through folder
        for file in files:
            if file.endswith(".wav"):
                emotion = int(file.split("-")[2])  # extract emotion label
                path = os.path.join(subdir, file)  # create file path
                data, _ = librosa.load(path, sr=sample_rate, duration=2.5, offset=0.6)  # load audio
                features = extract_features(data, sample_rate)  # extract features
                x.append(features)  # append features
                y.append(emotion)  # append labels
    return np.array(x), np.array(y)  # convert lists to arrays

folder_path = "speech"  # folder containing audio files
sample_rate = 22050  # sample rate

# get features and labels from audio files
x, y = get_features(folder_path, sample_rate)

encoder = OneHotEncoder()  # one-hot encoder
y_encoded = encoder.fit_transform(y.reshape(-1, 1)).toarray()  # encode labels

# split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y_encoded, test_size=0.2, random_state=42)

scaler = StandardScaler()  # standard scaler
x_train_scaled = scaler.fit_transform(x_train)  # scale training data
x_test_scaled = scaler.transform(x_test)  # scale testing data

x_train_reshaped = np.expand_dims(x_train_scaled, axis=2)  # reshape training data
x_test_reshaped = np.expand_dims(x_test_scaled, axis=2)  # reshape testing data

model = Sequential()  # create sequential model
model.add(Conv1D(256, kernel_size=5, strides=1, padding='same', activation='relu', input_shape=(x_train.shape[1], 1)))  # 1st Convolutional layer
model.add(MaxPooling1D(pool_size=5, strides=2, padding='same'))  # 1st Max pooling layer
model.add(Conv1D(256, kernel_size=5, strides=1, padding='same', activation='relu'))  # 2nd Convolutional layer
model.add(MaxPooling1D(pool_size=5, strides=2, padding='same'))  # 2nd Max pooling layer
model.add(Conv1D(128, kernel_size=5, strides=1, padding='same', activation='relu'))  # 3rd Convolutional layer
model.add(MaxPooling1D(pool_size=5, strides=2, padding='same'))  # 3rd Max pooling layer
model.add(Dropout(0.2))  # dropout layer
model.add(Conv1D(64, kernel_size=5, strides=1, padding='same', activation='relu'))  # 4th Convolutional layer
model.add(MaxPooling1D(pool_size=5, strides=2, padding='same'))  # 4th Max pooling layer
model.add(Flatten())  # flatten layer
model.add(Dense(units=32, activation='relu'))  # fully connected layer
model.add(Dropout(0.3))  # dropout layer
model.add(Dense(units=8, activation='softmax'))  # output layer
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])  # compile model

rlrp = ReduceLROnPlateau(monitor='loss', factor=0.4, verbose=0, patience=2, min_lr=0.0000001)  # learning rate reduction callback

# train the model
history = model.fit(x_train_reshaped, y_train, batch_size=64, epochs=50, validation_data=(x_test_reshaped, y_test), callbacks=[rlrp])

model.save("speech_emotion_detection_model.h5")  # save the complete model including architecture and trained weights

model_json = model.to_json()  # convert model architecture to JSON format
with open("speech_emotion_detection_model.json", "w") as json_file:
    json_file.write(model_json)  # save model architecture to JSON file for future model loading and reconstruction

with open('scaler.pickle', 'wb') as f:  # save scaler for audio feature scaling
    pickle.dump(scaler, f)  # standardScaler is used to standardize features, ensuring consistent scaling for training and prediction

with open('encoder.pickle', 'wb') as f:  # save encoder for emotion label encoding
    pickle.dump(encoder, f)  # OneHotEncoder is used to convert emotion labels into a suitable format for classification