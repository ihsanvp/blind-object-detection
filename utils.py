import cv2
import threading
import pyttsx3
import numpy as np
import tensorflow as tf


class AudioEngine:
    def __init__(self):
        self.last = None

    def say(self, data):
        if not self.last == data:
            thread = threading.Thread(target=AudioEngine.run, args=(data,))
            self.last = data
            if not thread.is_alive():
                thread.start() 

    @staticmethod
    def run(data):
        engine = pyttsx3.init()
        voices = engine.getProperty("voices")
        engine.setProperty("voice", voices[1].id)
        engine.setProperty("volume", 1.0)
        engine.setProperty("rate", 120)
        engine.say(data)
        engine.runAndWait()
        


class ObjectDetector:
    LABELS_PATH = "data/labels.txt"
    MODEL_PATH = "data/keras_model.h5"

    def __init__(self, match_threshold = 0):
        self.match_threshold = match_threshold
        self.labels = ObjectDetector.load_labels()
        self.model = tf.keras.models.load_model(ObjectDetector.MODEL_PATH, compile=False)

    def detect(self, image):
        processed_data = self.preprocess(image)
        predictions = self.model.predict(processed_data, verbose=0)
        prediction = np.squeeze(predictions)
        index = np.argmax(prediction)
        score = prediction[index]
        label = self.labels[index]

        if score > self.match_threshold:
            is_match = True
        else:
            is_match = False

        return is_match, label, score

    def preprocess(self, image):
        processed = cv2.resize(image, (224, 224), interpolation=cv2.INTER_CUBIC)
        processed = np.asarray(processed, dtype=np.float32)
        processed = np.expand_dims(processed, axis=0)
        processed = (processed / 127.5) - 1
        return processed
    
    @staticmethod
    def load_labels():
        with open(ObjectDetector.LABELS_PATH, "r") as f:
            labels_data = f.readlines()

        labels = list(map(lambda x: x.strip(), labels_data))
        return labels

    