# AI Health Diagnostic Platform

An AI-powered diagnostic platform for skin disease screening and pregnancy risk prediction.

## Overview

This platform addresses key healthcare challenges by:

1. **Integrating Skin Disease Screening**: Using machine learning to analyze skin images and identify early signs of conditions like melanoma, eczema, and psoriasis.

2. **Predicting Pregnancy Risks**: Leveraging machine learning models to assess patient data and predict potential risks during pregnancy.

3. **Enhancing Decision-Making**: Providing real-time, AI-driven insights to reduce diagnostic errors and aid healthcare providers.

4. **Offering Accessibility**: Expanding diagnostic capabilities to underserved regions.

## Features

### Skin Disease Screening

- Precisely identifies disorders including melanoma, eczema, and psoriasis
- Provides severity analysis (mild, moderate, severe)
- Generates actionable reports with recommended next steps
- Uses a deep learning model trained on dermatological datasets

### Pregnancy Risk Prediction

- Calculates personalized risk scores using medical history and vital signs
- Provides early alerts for high-risk conditions like preeclampsia
- Offers ongoing surveillance and actionable insights
- Based on the maternal health risk dataset

## Project Structure

- `ai_health_diagnostic.py`: Main implementation with both diagnostic systems
- `skin_disease_dataset_info.py`: Information about skin disease datasets
- `pregnancy_risk_dataset_info.py`: Information about the pregnancy risk dataset

## Requirements

- Python 3.8+
- TensorFlow 2.x
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- OpenCV (cv2)

## Getting Started

1. Clone this repository
2. Install the required dependencies:
   ```
   pip install tensorflow scikit-learn pandas numpy matplotlib seaborn opencv-python
   ```
3. Run the demo:
   ```
   python ai_health_diagnostic.py
   ```

## Dataset Information

### Skin Disease Datasets

For training the skin disease classification model, we recommend using:

- **HAM10000 Dataset**: 10,000+ dermatoscopic images of pigmented skin lesions
- **Dermnet Dataset**: 23,000+ images of skin diseases
- **ISIC Archive**: Large collection of dermoscopic images
- **SD-198 Dataset**: 6,584 clinical images of 198 skin diseases

See `skin_disease_dataset_info.py` for more details.

### Pregnancy Risk Dataset

For training the pregnancy risk prediction model, we use:

- **Maternal Health Risk Dataset**: Contains data on age, blood pressure, blood glucose, body temperature, and heart rate, with risk level classifications.

See `pregnancy_risk_dataset_info.py` for more details.

## Model Architecture

### Skin Disease Classification

- Uses MobileNetV2 as the base model (pre-trained on ImageNet)
- Multi-output model for disease classification and severity assessment
- Data augmentation for improved generalization

### Pregnancy Risk Prediction

- Random Forest Classifier
- Feature scaling with StandardScaler
- Trained on maternal health risk dataset

## Future Improvements

- Add more disease categories to the skin disease classifier
- Incorporate more features for pregnancy risk prediction
- Implement explainable AI techniques for better interpretability
- Add cloud storage integration for secure data management

## Disclaimer

This system is intended to assist healthcare professionals and should not replace professional medical advice. Always consult with a qualified healthcare provider for diagnosis and treatment.
