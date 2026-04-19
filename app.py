"""
Animal Disease Prediction - Flask Web Application
Serves the ML model via a REST API with a beautiful frontend.
"""

from flask import Flask, render_template, request, jsonify
import pickle
import json
import numpy as np
import os

app = Flask(__name__)

# ============================================================
# Load Model & Metadata
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = pickle.load(open(os.path.join(BASE_DIR, "animal_disease_model.pkl"), "rb"))
label_encoders = pickle.load(open(os.path.join(BASE_DIR, "label_encoders.pkl"), "rb"))

with open(os.path.join(BASE_DIR, "label_encoder_classes.json"), "r") as f:
    encoder_classes = json.load(f)

with open(os.path.join(BASE_DIR, "feature_columns.json"), "r") as f:
    feature_columns = json.load(f)

with open(os.path.join(BASE_DIR, "disease_mapping.json"), "r") as f:
    disease_mapping = json.load(f)


# ============================================================
# Disease Info Database (for rich results)
# ============================================================
DISEASE_INFO = {
    "Bovine Tuberculosis": {
        "severity": "High",
        "description": "A chronic bacterial disease caused by Mycobacterium bovis, primarily affecting cattle but can spread to other species.",
        "treatment": "Quarantine, testing and culling of infected animals, antibiotics in some cases.",
        "icon": "🐄"
    },
    "Bovine Respiratory Disease": {
        "severity": "High",
        "description": "A complex of diseases affecting the respiratory tract of cattle, caused by multiple pathogens.",
        "treatment": "Antibiotics, anti-inflammatory drugs, supportive care, vaccination prevention.",
        "icon": "🐄"
    },
    "Equine Influenza": {
        "severity": "Medium",
        "description": "A highly contagious respiratory disease of horses caused by influenza A viruses.",
        "treatment": "Rest, supportive care, anti-inflammatory medication, vaccination prevention.",
        "icon": "🐴"
    },
    "Canine Parvovirus": {
        "severity": "Critical",
        "description": "A highly contagious viral disease in dogs that causes severe gastrointestinal illness.",
        "treatment": "Hospitalization, IV fluids, anti-nausea medication, antibiotics for secondary infections.",
        "icon": "🐕"
    },
    "Caprine Arthritis Encephalitis": {
        "severity": "Medium",
        "description": "A viral disease of goats causing arthritis in adults and encephalitis in young kids.",
        "treatment": "No cure available. Management focuses on supportive care and preventing transmission.",
        "icon": "🐐"
    },
    "Canine Distemper": {
        "severity": "Critical",
        "description": "A serious viral illness affecting dogs that attacks the respiratory, gastrointestinal, and nervous systems.",
        "treatment": "Supportive care, IV fluids, anti-seizure medication, antibiotics for secondary infections.",
        "icon": "🐕"
    },
    "Scrapie": {
        "severity": "High",
        "description": "A fatal degenerative disease affecting the nervous system of sheep and goats (prion disease).",
        "treatment": "No treatment available. Prevention through genetic testing and selective breeding.",
        "icon": "🐑"
    },
    "Swine Influenza": {
        "severity": "Medium",
        "description": "A respiratory disease of pigs caused by type A influenza viruses.",
        "treatment": "Supportive care, maintaining hydration, anti-inflammatory drugs, vaccination.",
        "icon": "🐷"
    },
    "Kennel Cough": {
        "severity": "Low",
        "description": "A highly contagious respiratory disease in dogs caused by Bordetella bronchiseptica and other agents.",
        "treatment": "Rest, cough suppressants, antibiotics if bacterial, usually self-limiting.",
        "icon": "🐕"
    },
    "Equine Infectious Anemia": {
        "severity": "High",
        "description": "A viral disease of horses transmitted by blood-sucking insects, causing recurring fever and anemia.",
        "treatment": "No cure. Infected horses must be quarantined or euthanized to prevent spread.",
        "icon": "🐴"
    }
}

# Default info for unknown diseases
DEFAULT_INFO = {
    "severity": "Unknown",
    "description": "Disease information not available in our database.",
    "treatment": "Please consult a licensed veterinarian for proper diagnosis and treatment.",
    "icon": "🏥"
}


@app.route("/")
def home():
    """Serve the main prediction page."""
    return render_template("index.html", encoder_classes=encoder_classes)


@app.route("/api/metadata", methods=["GET"])
def get_metadata():
    """Return form metadata for dynamic frontend rendering."""
    return jsonify({
        "encoder_classes": encoder_classes,
        "feature_columns": feature_columns,
        "diseases": list(disease_mapping.values())
    })


@app.route("/predict", methods=["POST"])
def predict():
    """Accept form data, encode, predict, and return result."""
    try:
        data = request.get_json()

        # Build feature vector in correct column order
        features = []
        for col in feature_columns:
            value = data.get(col)

            if col in label_encoders:
                le = label_encoders[col]
                try:
                    encoded_val = le.transform([str(value)])[0]
                except ValueError:
                    # Handle unseen labels gracefully
                    encoded_val = 0
                features.append(encoded_val)
            else:
                # Numeric column
                features.append(float(value))

        # Predict
        prediction = model.predict([features])[0]

        # Decode prediction back to disease name
        disease_name = disease_mapping.get(str(int(prediction)), f"Disease #{prediction}")

        # Get disease info
        info = DISEASE_INFO.get(disease_name, DEFAULT_INFO)

        return jsonify({
            "success": True,
            "prediction": disease_name,
            "severity": info["severity"],
            "description": info["description"],
            "treatment": info["treatment"],
            "icon": info["icon"]
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400


if __name__ == "__main__":
    print("[*] Animal Disease Prediction Server Starting...")
    print("    Open http://127.0.0.1:5000 in your browser")
    app.run(debug=True, port=5000)
