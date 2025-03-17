import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import (
    Dense, Conv2D, MaxPooling2D, Flatten, Dropout, 
    BatchNormalization, Input, GlobalMaxPooling2D
)
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

import cv2
import pickle
import warnings
warnings.filterwarnings('ignore')
from flask import Flask, request, jsonify
from twilio.rest import Client

app = Flask(__name__)

# Twilio credentials
TWILIO_ACCOUNT_SID = "AC7c045454b692b27a3fc9762e416f6688"
TWILIO_AUTH_TOKEN = "3c23886053e9991ac44a7ce6daa3983e"
TWILIO_PHONE_NUMBER = ""+17853903931
DEMO_PHONE_NUMBER = "6281520694"  # Replace with your number

client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

@app.route('/call-hospital', methods=['POST'])
def call_hospital():
    try:
        call = client.calls.create(
            to=DEMO_PHONE_NUMBER,
            from_=TWILIO_PHONE_NUMBER,
            url="http://demo.twilio.com/docs/voice.xml"  # Twilio XML for voice message
        )
        return jsonify({"message": "Call initiated successfully!", "call_sid": call.sid}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500



class SkinDiseaseClassifier:
    def __init__(self):
        self.model = None
        self.class_names = ['Melanoma', 'Eczema', 'Psoriasis', 'Normal']
        self.severity_levels = ['Mild', 'Moderate', 'Severe']
        
    def build_model(self, input_shape=(224, 224, 3)):
        """Build and compile the CNN model for skin disease classification"""
        # Use MobileNetV2 as base model (lightweight and efficient)
        base_model = MobileNetV2(
            input_shape=input_shape,
            include_top=False,
            weights='imagenet'
        )
        
        # Freeze the base model layers
        base_model.trainable = False
        
        # Create the model architecture
        inputs = Input(shape=input_shape)
        x = base_model(inputs, training=False)
        x = GlobalMaxPooling2D()(x)  # ✅ Fixed Import Issue
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(64, activation='relu')(x)
        
        # Main classification output
        disease_output = Dense(len(self.class_names), activation='softmax', name='disease')(x)
        
        # Severity classification output (only if disease is detected)
        severity_output = Dense(len(self.severity_levels), activation='softmax', name='severity')(x)
        
        # Create multi-output model
        self.model = Model(inputs=inputs, outputs=[disease_output, severity_output])
        
        # Compile the model
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss={'disease': 'categorical_crossentropy', 'severity': 'categorical_crossentropy'},
            metrics={'disease': 'accuracy', 'severity': 'accuracy'}
        )

# ✅ Function to run the classifier
def main():
    print("\nAI Health Diagnostic Platform Demo")
    print("="*34)
    
    print("\n=== Skin Disease Classification Demo ===")
    
    classifier = SkinDiseaseClassifier()
    classifier.build_model()
    print("Model successfully built! ✅")

if __name__ == "__main__":
    main()
