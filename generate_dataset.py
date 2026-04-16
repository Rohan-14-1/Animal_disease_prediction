"""
Generate a synthetic CSV dataset matching the notebook's schema.
This recreates 'cleaned_animal_disease_prediction.csv' so the
training script can run without the original Kaggle file.
"""

import pandas as pd
import numpy as np
import random

random.seed(42)
np.random.seed(42)

# ---- Schema from notebook ----
ANIMAL_BREEDS = {
    "Dog": ["Labrador", "Beagle", "German Shepherd", "Bulldog", "Poodle",
            "Golden Retriever", "Rottweiler", "Husky", "Dalmatian", "Boxer"],
    "Cat": ["Siamese", "Persian", "Maine Coon", "Bengal", "Ragdoll",
            "British Shorthair", "Abyssinian", "Sphynx"],
    "Cow": ["Holstein", "Jersey", "Angus", "Hereford", "Brahman", "Charolais"],
    "Horse": ["Thoroughbred", "Arabian", "Quarter Horse", "Mustang",
              "Clydesdale", "Appaloosa"],
    "Goat": ["Boer", "Alpine", "Nubian", "Saanen", "Angora", "LaMancha"],
    "Sheep": ["Merino", "Suffolk", "Dorper", "Romney", "Hampshire", "Texel"],
    "Pig": ["Yorkshire", "Duroc", "Berkshire", "Hampshire Pig", "Landrace", "Pietrain"]
}

SYMPTOMS = [
    "Fever", "Lethargy", "Appetite Loss", "Vomiting", "Coughing",
    "Sneezing", "Eye Discharge", "Nasal Discharge", "Diarrhea",
    "Labored Breathing", "Skin Lesions", "Lameness", "Swelling",
    "Weight Loss", "Dehydration"
]

DURATIONS = [
    "1 day", "2 days", "3 days", "4 days", "5 days",
    "1 week", "2 weeks", "3 weeks", "1 month", "2 months",
    "3 months", "6 months"
]

BODY_TEMPS = [
    "37.5°C", "37.8°C", "38.0°C", "38.2°C", "38.5°C", "38.7°C",
    "38.9°C", "39.0°C", "39.2°C", "39.5°C", "39.8°C", "40.0°C",
    "40.1°C", "40.5°C", "41.0°C"
]

# Disease → animal type mapping
DISEASE_ANIMALS = {
    "Canine Parvovirus":              ["Dog"],
    "Canine Distemper":               ["Dog"],
    "Kennel Cough":                   ["Dog"],
    "Bovine Tuberculosis":            ["Cow"],
    "Bovine Respiratory Disease":     ["Cow"],
    "Foot and Mouth Disease":         ["Cow", "Goat", "Sheep", "Pig"],
    "Equine Influenza":               ["Horse"],
    "Equine Infectious Anemia":       ["Horse"],
    "Caprine Arthritis Encephalitis": ["Goat"],
    "Scrapie":                        ["Sheep", "Goat"],
    "Swine Influenza":                ["Pig"],
    "Upper Respiratory Infection":    ["Cat"],
    "Fungal Infection":               ["Cat", "Dog"],
    "Gastroenteritis":                ["Dog", "Cat"],
    "Parvovirus":                     ["Dog"],
    "Mange":                          ["Dog", "Cat"],
    "Rabies":                         ["Dog", "Cat"],
    "Mastitis":                       ["Cow", "Goat"],
    "Coccidiosis":                    ["Goat", "Sheep"],
    "Strangles":                      ["Horse"],
}

WEIGHT_RANGES = {
    "Dog": (5, 50),
    "Cat": (2.5, 8),
    "Cow": (300, 750),
    "Horse": (350, 700),
    "Goat": (25, 90),
    "Sheep": (30, 120),
    "Pig": (50, 300),
}

def gen_row(disease):
    animal_type = random.choice(DISEASE_ANIMALS[disease])
    breed = random.choice(ANIMAL_BREEDS[animal_type])
    age = random.randint(1, 14)
    gender = random.choice(["Male", "Female"])
    w_lo, w_hi = WEIGHT_RANGES[animal_type]
    weight = round(random.uniform(w_lo, w_hi), 1)

    syms = random.sample(SYMPTOMS, 4)
    duration = random.choice(DURATIONS)

    yes_no = lambda: random.choice(["Yes", "No"])

    return {
        "Animal_Type": animal_type,
        "Breed": breed,
        "Age": age,
        "Gender": gender,
        "Weight": weight,
        "Symptom_1": syms[0],
        "Symptom_2": syms[1],
        "Symptom_3": syms[2],
        "Symptom_4": syms[3],
        "Duration": duration,
        "Appetite_Loss": yes_no(),
        "Vomiting": yes_no(),
        "Diarrhea": yes_no(),
        "Coughing": yes_no(),
        "Labored_Breathing": yes_no(),
        "Lameness": yes_no(),
        "Skin_Lesions": yes_no(),
        "Nasal_Discharge": yes_no(),
        "Eye_Discharge": yes_no(),
        "Body_Temperature": random.choice(BODY_TEMPS),
        "Heart_Rate": random.randint(60, 160),
        "Disease_Prediction": disease,
    }


rows = []
for disease in DISEASE_ANIMALS:
    n = random.randint(15, 30)
    for _ in range(n):
        rows.append(gen_row(disease))

random.shuffle(rows)
df = pd.DataFrame(rows)

df.to_csv("cleaned_animal_disease_prediction.csv", index=False)
print(f"[OK] Generated {len(df)} rows -> cleaned_animal_disease_prediction.csv")
print(f"   Diseases: {df['Disease_Prediction'].nunique()}")
print(df['Disease_Prediction'].value_counts())
