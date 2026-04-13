# VetAI - Animal Disease Prediction System 🐾

A full-stack web application that predicts potential animal diseases based on symptoms, clinical signs, and vital information. Powered by a Machine Learning model (Random Forest Classifier).

## Features

- **AI-Powered Diagnostics**: Utilizes a trained Random Forest model to predict diseases from input symptoms.
- **Comprehensive Data Entry**: A 4-step diagnostic form covering basic info, symptoms, clinical signs (Yes/No toggles), and vitals.
- **Rich Results Dashboard**: Displays predicted disease, severity level, description, and treatment recommendations.
- **Premium UI**: A modern, responsive dark theme with glassmorphism effects, animated background particles, and smooth transitions.
- **RESTful API**: Flask backend serving model predictions via JSON endpoints.

## Tech Stack

- **Backend / Machine Learning**:
  - Python 3
  - Flask (Web Framework)
  - Scikit-Learn (Model Training & Inference)
  - Pandas & NumPy (Data Processing)
- **Frontend**:
  - HTML5
  - CSS3 (Custom Properties, Flexbox/Grid, Animations)
  - Vanilla JavaScript (Fetch API, DOM manipulation)

## Project Structure

```text
animal_disease_prediction/
├── animal_disease_prediction.py  # Script for training the ML model
├── app.py                        # Flask server and API endpoints
├── generate_dataset.py           # Script to generate a synthetic dataset
├── requirements.txt              # Python dependencies
├── animal_disease_model.pkl      # Saved trained model (generated)
├── label_encoders.pkl            # Saved encoders for inference (generated)
├── static/
│   ├── style.css                 # Frontend styling
│   └── script.js                 # Frontend logic and API calls
└── templates/
    └── index.html                # Main UI template
```

## Setup & Installation

### 1. Prerequisites
Make sure you have Python 3.8+ installed.

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Generate Dataset (Optional)
If you do not have the original `cleaned_animal_disease_prediction.csv`, you can generate a synthetic one that matches the expected schema:
```bash
python generate_dataset.py
```

### 4. Train the Model
Run the training script to build the model and generate the necessary `.pkl` and `.json` artifacts:
```bash
python animal_disease_prediction.py
```

### 5. Start the Web Server
Launch the Flask application:
```bash
python app.py
```

### 6. Use the App
Open your web browser and navigate to:
**http://127.0.0.1:5000**

## Disclaimer
*This is an AI-based educational tool and should **not** replace professional veterinary consultation. Always consult a licensed veterinarian for actual medical advice.*
