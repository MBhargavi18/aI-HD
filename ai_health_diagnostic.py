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
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization, Input
from tensorflow.keras.applications import MobileNetV2, InceptionV3
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import GlobalMaxPooling2D


import cv2
import pickle
import warnings
warnings.filterwarnings('ignore')

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
        x = GlobalMaxPooling2D()(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(64, activation='relu')(x)
        
        # Main classification output
        disease_output = Dense(len(self.class_names), activation='softmax', name='disease')(x)
        
        # Severity classification output (only if disease is detected)
        severity_output = Dense(len(self.severity_levels), activation='softmax', name='severity')(x)
        
        # Create multi-output model
        model = Model(inputs=inputs, outputs=[disease_output, severity_output])
        
        # Compile the model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss={
                'disease': 'categorical_crossentropy',
                'severity': 'categorical_crossentropy'
            },
            metrics={
                'disease': 'accuracy',
                'severity': 'accuracy'
            }
        )
        
        self.model = model
        return model
    
    def preprocess_image(self, image_path):
        """Preprocess an image for model prediction"""
        # Read and resize image
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        
        # Normalize pixel values
        img = img / 255.0
        
        # Expand dimensions to match model input shape
        img = np.expand_dims(img, axis=0)
        
        return img
    
    def train(self, train_dir, validation_dir, epochs=20, batch_size=32):
        """Train the model on a dataset of skin disease images"""
        if self.model is None:
            self.build_model()
            
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        # Only rescaling for validation
        validation_datagen = ImageDataGenerator(rescale=1./255)
        
        # Flow images from directories
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(224, 224),
            batch_size=batch_size,
            class_mode='categorical'
        )
        
        validation_generator = validation_datagen.flow_from_directory(
            validation_dir,
            target_size=(224, 224),
            batch_size=batch_size,
            class_mode='categorical'
        )
        
        # Callbacks for training
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
            ModelCheckpoint('skin_disease_model.h5', save_best_only=True)
        ]
        
        # Train the model
        history = self.model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // batch_size,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=validation_generator.samples // batch_size,
            callbacks=callbacks
        )
        
        return history
    
    def predict(self, image_path):
        """Predict skin disease and severity from an image"""
        if self.model is None:
            raise ValueError("Model not trained or loaded. Please train or load a model first.")
            
        # Preprocess the image
        img = self.preprocess_image(image_path)
        
        # Make prediction
        disease_pred, severity_pred = self.model.predict(img)
        
        # Get the predicted class and confidence
        disease_idx = np.argmax(disease_pred[0])
        disease_confidence = disease_pred[0][disease_idx]
        disease_name = self.class_names[disease_idx]
        
        severity_idx = np.argmax(severity_pred[0])
        severity_level = self.severity_levels[severity_idx]
        severity_confidence = severity_pred[0][severity_idx]
        
        # Create result dictionary
        result = {
            'disease': {
                'name': disease_name,
                'confidence': float(disease_confidence)
            },
            'severity': {
                'level': severity_level,
                'confidence': float(severity_confidence)
            }
        }
        
        return result
    
    def generate_report(self, prediction_result):
        """Generate a user-friendly report based on prediction results"""
        disease = prediction_result['disease']['name']
        disease_conf = prediction_result['disease']['confidence'] * 100
        severity = prediction_result['severity']['level']
        
        report = {
            'diagnosis': disease,
            'confidence': f"{disease_conf:.1f}%",
            'severity': severity,
            'description': self._get_disease_description(disease),
            'recommendations': self._get_recommendations(disease, severity)
        }
        
        return report
    
    def _get_disease_description(self, disease):
        """Return description for the detected skin disease"""
        descriptions = {
            'Melanoma': "Melanoma is a serious form of skin cancer that begins in cells known as melanocytes. While it is less common than other types of skin cancer, it causes the majority of skin cancer deaths.",
            'Eczema': "Eczema (atopic dermatitis) is a condition that causes your skin to become dry, itchy, red and cracked. It is common in children but can occur at any age.",
            'Psoriasis': "Psoriasis is a skin disorder that causes skin cells to multiply up to 10 times faster than normal. This makes the skin build up into bumpy red patches covered with white scales.",
            'Normal': "No concerning skin condition detected. Your skin appears healthy based on the provided image."
        }
        
        return descriptions.get(disease, "No description available.")
    
    def _get_recommendations(self, disease, severity):
        """Return recommendations based on disease and severity"""
        if disease == 'Normal':
            return [
                "Continue with regular skin care routine",
                "Protect your skin from sun exposure",
                "Perform regular self-examinations"
            ]
            
        recommendations = {
            'Melanoma': {
                'Mild': [
                    "Schedule an appointment with a dermatologist within 1-2 weeks",
                    "Avoid sun exposure to the affected area",
                    "Take photographs to monitor any changes"
                ],
                'Moderate': [
                    "See a dermatologist urgently (within days)",
                    "Prepare for possible biopsy procedure",
                    "Avoid sun exposure completely"
                ],
                'Severe': [
                    "Seek immediate medical attention",
                    "This requires urgent specialist assessment",
                    "Prepare for possible surgical intervention"
                ]
            },
            'Eczema': {
                'Mild': [
                    "Use over-the-counter moisturizers regularly",
                    "Avoid known triggers (certain soaps, detergents, etc.)",
                    "Take short, lukewarm showers instead of hot baths"
                ],
                'Moderate': [
                    "Consult with a dermatologist for prescription treatments",
                    "Consider topical corticosteroids",
                    "Keep the affected area clean and moisturized"
                ],
                'Severe': [
                    "Seek dermatologist care promptly",
                    "May require prescription-strength medications or phototherapy",
                    "Watch for signs of infection (increased pain, pus, etc.)"
                ]
            },
            'Psoriasis': {
                'Mild': [
                    "Use over-the-counter salicylic acid products",
                    "Apply moisturizer regularly",
                    "Get moderate sun exposure (10-15 minutes daily)"
                ],
                'Moderate': [
                    "Consult with a dermatologist for prescription treatments",
                    "Consider topical treatments containing corticosteroids or vitamin D",
                    "Avoid triggers like stress, alcohol, and smoking"
                ],
                'Severe': [
                    "See a dermatologist for advanced treatment options",
                    "May require systemic medications or biologics",
                    "Consider phototherapy under medical supervision"
                ]
            }
        }
        
        return recommendations.get(disease, {}).get(severity, ["Consult with a healthcare professional for proper diagnosis and treatment"])
    
    def save_model(self, filepath):
        """Save the trained model to disk"""
        if self.model is None:
            raise ValueError("No model to save. Train a model first.")
        self.model.save(filepath)
        
    def load_model(self, filepath):
        """Load a trained model from disk"""
        self.model = load_model(filepath)


class PregnancyRiskPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.risk_levels = ['low risk', 'mid risk', 'high risk']
        
    def preprocess_data(self, data):
        """Preprocess the input data for prediction"""
        # Create a copy to avoid modifying the original data
        df = data.copy()
        
        # Feature scaling
        if self.scaler is None:
            self.scaler = StandardScaler()
            df_scaled = self.scaler.fit_transform(df)
        else:
            df_scaled = self.scaler.transform(df)
            
        return df_scaled
    
    def build_model(self):
        """Build and compile the model for pregnancy risk prediction"""
        # Using Random Forest for this task
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        return self.model
    
    def train(self, X, y):
        """Train the model on pregnancy data"""
        if self.model is None:
            self.build_model()
            
        # Preprocess the data
        X_scaled = self.preprocess_data(X)
        
        # Train the model
        self.model.fit(X_scaled, y)
        
        # Evaluate on training data
        y_pred = self.model.predict(X_scaled)
        accuracy = accuracy_score(y, y_pred)
        
        print(f"Training accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y, y_pred, target_names=self.risk_levels))
        
        return self.model
    
    def predict(self, patient_data):
        """Predict pregnancy risk level for a patient"""
        if self.model is None:
            raise ValueError("Model not trained or loaded. Please train or load a model first.")
            
        # Ensure data is in the right format (DataFrame)
        if not isinstance(patient_data, pd.DataFrame):
            patient_data = pd.DataFrame([patient_data])
            
        # Preprocess the data
        patient_data_scaled = self.preprocess_data(patient_data)
        
        # Make prediction
        risk_idx = self.model.predict(patient_data_scaled)[0]
        risk_probabilities = self.model.predict_proba(patient_data_scaled)[0]
        
        # Get the predicted risk level and confidence
        risk_level = self.risk_levels[risk_idx]
        confidence = risk_probabilities[risk_idx]
        
        # Create result dictionary
        result = {
            'risk_level': risk_level,
            'confidence': float(confidence),
            'probabilities': {
                level: float(prob) for level, prob in zip(self.risk_levels, risk_probabilities)
            }
        }
        
        return result
    
    def generate_report(self, prediction_result, patient_data):
        """Generate a user-friendly report based on prediction results"""
        risk_level = prediction_result['risk_level']
        confidence = prediction_result['confidence'] * 100
        
        # Extract patient data for the report
        age = patient_data.get('Age', 'N/A')
        systolic_bp = patient_data.get('SystolicBP', 'N/A')
        diastolic_bp = patient_data.get('DiastolicBP', 'N/A')
        blood_glucose = patient_data.get('BS', 'N/A')
        body_temp = patient_data.get('BodyTemp', 'N/A')
        heart_rate = patient_data.get('HeartRate', 'N/A')
        
        report = {
            'risk_assessment': risk_level.upper(),
            'confidence': f"{confidence:.1f}%",
            'patient_summary': {
                'Age': age,
                'Blood Pressure': f"{systolic_bp}/{diastolic_bp} mmHg",
                'Blood Glucose': f"{blood_glucose} mmol/L",
                'Body Temperature': f"{body_temp} °C",
                'Heart Rate': f"{heart_rate} bpm"
            },
            'recommendations': self._get_recommendations(risk_level, patient_data),
            'follow_up': self._get_follow_up(risk_level)
        }
        
        return report
    
    def _get_recommendations(self, risk_level, patient_data):
        """Return recommendations based on risk level and patient data"""
        # Base recommendations for all patients
        base_recommendations = [
            "Maintain a balanced diet rich in fruits, vegetables, and whole grains",
            "Stay hydrated by drinking plenty of water throughout the day",
            "Get regular moderate exercise as approved by your healthcare provider"
        ]
        
        # Risk-specific recommendations
        risk_recommendations = {
            'low risk': [
                "Continue with routine prenatal check-ups",
                "Monitor fetal movements daily",
                "Take prenatal vitamins as prescribed"
            ],
            'mid risk': [
                "Increase frequency of prenatal visits as advised by your doctor",
                "Monitor blood pressure regularly at home if possible",
                "Limit sodium intake and avoid processed foods",
                "Be vigilant about reporting any new symptoms to your healthcare provider"
            ],
            'high risk': [
                "Urgent consultation with an obstetrician is recommended",
                "Strict monitoring of blood pressure and other vital signs",
                "Consider hospital evaluation if experiencing severe headaches, vision changes, or upper abdominal pain",
                "Prepare for possible early delivery or additional interventions",
                "Reduce physical activity and consider bed rest as advised by your doctor"
            ]
        }
        
        # Add specific recommendations based on patient data
        specific_recommendations = []
        
        # Check blood pressure
        systolic_bp = patient_data.get('SystolicBP', 0)
        diastolic_bp = patient_data.get('DiastolicBP', 0)
        
        if systolic_bp > 140 or diastolic_bp > 90:
            specific_recommendations.append("Your blood pressure is elevated. Reduce salt intake and monitor blood pressure daily.")
            
        # Check blood glucose
        blood_glucose = patient_data.get('BS', 0)
        if blood_glucose > 7.8:
            specific_recommendations.append("Your blood glucose level is elevated. Limit sugar intake and consider consulting with a dietitian.")
            
        # Check heart rate
        heart_rate = patient_data.get('HeartRate', 0)
        if heart_rate > 100:
            specific_recommendations.append("Your heart rate is elevated. Ensure adequate rest and avoid strenuous activities.")
            
        # Combine all recommendations
        all_recommendations = base_recommendations + risk_recommendations.get(risk_level, []) + specific_recommendations
        
        return all_recommendations
    
    def _get_follow_up(self, risk_level):
        """Return follow-up recommendations based on risk level"""
        follow_ups = {
            'low risk': "Schedule your next routine prenatal visit in 4 weeks.",
            'mid risk': "Schedule a follow-up appointment within 1-2 weeks to reassess your condition.",
            'high risk': "Immediate follow-up with a maternal-fetal medicine specialist is recommended within the next 24-48 hours."
        }
        
        return follow_ups.get(risk_level, "Consult with your healthcare provider for appropriate follow-up timing.")
    
    def save_model(self, filepath):
        """Save the trained model to disk"""
        if self.model is None:
            raise ValueError("No model to save. Train a model first.")
            
        # Save both the model and the scaler
        with open(filepath, 'wb') as f:
            pickle.dump({'model': self.model, 'scaler': self.scaler}, f)
        
    def load_model(self, filepath):
        """Load a trained model from disk"""
        with open(filepath, 'rb') as f:
            saved_data = pickle.load(f)
            
        self.model = saved_data['model']
        self.scaler = saved_data['scaler']


# Example usage for Skin Disease Classification
def demo_skin_disease_classifier():
    print("=== Skin Disease Classification Demo ===")
    
    # Initialize the classifier
    classifier = SkinDiseaseClassifier()
    
    # Build the model
    classifier.build_model()
    
    # In a real scenario, you would train the model or load a pre-trained model
    # classifier.train('path/to/train_data', 'path/to/validation_data')
    # classifier.load_model('skin_disease_model.h5')
    
    # For demo purposes, we'll simulate a prediction
    print("\nSimulating prediction for a skin image...")
    
    # Simulated prediction result
    simulated_result = {
        'disease': {
            'name': 'Eczema',
            'confidence': 0.89
        },
        'severity': {
            'level': 'Moderate',
            'confidence': 0.76
        }
    }
    
    # Generate report
    report = classifier.generate_report(simulated_result)
    
    # Display the report
    print("\n=== Skin Disease Analysis Report ===")
    print(f"Diagnosis: {report['diagnosis']} (Confidence: {report['confidence']})")
    print(f"Severity: {report['severity']}")
    print("\nDescription:")
    print(report['description'])
    
    print("\nRecommendations:")
    for i, rec in enumerate(report['recommendations'], 1):
        print(f"{i}. {rec}")


# Example usage for Pregnancy Risk Prediction
def demo_pregnancy_risk_predictor():
    print("\n\n=== Pregnancy Risk Prediction Demo ===")
    
    # Load the maternal health risk dataset
    # In a real scenario, you would download the dataset from the GitHub repository
    # For demo purposes, we'll create a sample dataset
    
    print("\nCreating sample maternal health dataset...")
    
    # Sample data (based on the maternal health risk dataset structure)
    data = {
        'Age': [25, 32, 28, 40, 35, 22, 31, 29, 36, 27],
        'SystolicBP': [120, 140, 130, 160, 150, 110, 125, 135, 145, 115],
        'DiastolicBP': [80, 90, 85, 100, 95, 75, 82, 88, 92, 78],
        'BS': [6.5, 7.8, 7.0, 9.5, 8.2, 6.0, 6.8, 7.2, 8.0, 6.2],
        'BodyTemp': [37.0, 37.5, 37.2, 38.0, 37.8, 36.8, 37.1, 37.3, 37.6, 36.9],
        'HeartRate': [75, 88, 80, 95, 90, 72, 78, 82, 85, 76],
        'RiskLevel': [0, 1, 0, 2, 1, 0, 0, 1, 2, 0]  # 0: low risk, 1: mid risk, 2: high risk
    }
    
    df = pd.DataFrame(data)
    
    # Split the data
    X = df.drop('RiskLevel', axis=1)
    y = df['RiskLevel']
    
    # Initialize the predictor
    predictor = PregnancyRiskPredictor()
    
    # Build and train the model
    predictor.build_model()
    predictor.train(X, y)
    
    # Simulate a new patient
    print("\nSimulating prediction for a new patient...")
    
    new_patient = {
        'Age': 33,
        'SystolicBP': 142,
        'DiastolicBP': 92,
        'BS': 7.9,
        'BodyTemp': 37.6,
        'HeartRate': 89
    }
    
    # Convert to DataFrame
    new_patient_df = pd.DataFrame([new_patient])
    
    # Make prediction
    prediction = predictor.predict(new_patient_df)
    
    # Generate report
    report = predictor.generate_report(prediction, new_patient)
    
    # Display the report
    print("\n=== Pregnancy Risk Assessment Report ===")
    print(f"Risk Level: {report['risk_assessment']} (Confidence: {report['confidence']})")
    
    print("\nPatient Summary:")
    for key, value in report['patient_summary'].items():
        print(f"{key}: {value}")
    
    print("\nRecommendations:")
    for i, rec in enumerate(report['recommendations'], 1):
        print(f"{i}. {rec}")
    
    print("\nFollow-up:")
    print(report['follow_up'])


# Main function to demonstrate both systems
def main():
    print("AI Health Diagnostic Platform Demo")
    print("==================================")
    
    # Demonstrate skin disease classification
    demo_skin_disease_classifier()
    
    # Demonstrate pregnancy risk prediction
    demo_pregnancy_risk_predictor()
    
    print("\n==================================")
    print("Demo completed. In a real application, you would:")
    print("1. Train models on large, diverse datasets")
    print("2. Implement a user interface for image upload and data entry")
    print("3. Set up secure data storage and handling")
    print("4. Integrate with healthcare systems for follow-up")


if __name__ == "__main__":
    main()
