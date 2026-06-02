import os
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ── Training data ──────────────────────────────────────────
# Features: [file_size_kb, has_images, has_tables, is_csv, is_pdf, is_image]
# Labels: 0=text/pdf, 1=tabular, 2=image

X = [
    # text/pdf samples
    [120, 0, 0, 0, 1, 0],
    [200, 0, 1, 0, 1, 0],
    [80,  0, 0, 0, 1, 0],
    [340, 0, 0, 0, 1, 0],
    [60,  0, 0, 0, 0, 0],
    [90,  0, 0, 0, 1, 0],

    # tabular/csv samples
    [50,  0, 1, 1, 0, 0],
    [30,  0, 1, 1, 0, 0],
    [75,  0, 1, 1, 0, 0],
    [20,  0, 1, 1, 0, 0],
    [100, 0, 1, 1, 0, 0],
    [45,  0, 1, 1, 0, 0],

    # image samples
    [500, 1, 0, 0, 0, 1],
    [800, 1, 0, 0, 0, 1],
    [300, 1, 0, 0, 0, 1],
    [150, 1, 0, 0, 0, 1],
    [600, 1, 0, 0, 0, 1],
    [420, 1, 0, 0, 0, 1],
]

y = [0,0,0,0,0,0, 1,1,1,1,1,1, 2,2,2,2,2,2]

# ── Train ──────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

print("✅ Classifier trained!")
print(classification_report(y_test, clf.predict(X_test)))

# ── Save model ─────────────────────────────────────────────
os.makedirs("models", exist_ok=True)
with open("models/classifier.pkl", "wb") as f:
    pickle.dump(clf, f)
print("✅ Model saved to models/classifier.pkl")

# ── Predict function ───────────────────────────────────────
def load_classifier():
    with open("models/classifier.pkl", "rb") as f:
        return pickle.load(f)

def classify_document(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    size_kb = os.path.getsize(file_path) / 1024

    is_csv   = 1 if ext == ".csv"              else 0
    is_pdf   = 1 if ext == ".pdf"              else 0
    is_image = 1 if ext in [".jpg", ".jpeg",
                             ".png", ".webp"]  else 0
    has_images  = is_image
    has_tables  = is_csv

    features = [[size_kb, has_images, has_tables,
                 is_csv, is_pdf, is_image]]

    model = load_classifier()
    prediction = model.predict(features)[0]

    labels = {0: "text/pdf", 1: "tabular", 2: "image"}
    result = labels[prediction]
    print(f"📄 {os.path.basename(file_path)} → classified as: {result}")
    return result

# ── Test ───────────────────────────────────────────────────
if __name__ == "__main__":
    classify_document("test.txt")