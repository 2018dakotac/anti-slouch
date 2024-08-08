import pandas as pd
import json
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Function to extract features from landmarks
def extract_features(landmarks):
    features = []
    for landmark in landmarks:
        features.extend([landmark.get('x', 0), landmark.get('y', 0), landmark.get('z', 0), landmark.get('visibility', 0)])
    return features

def process_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
        features = []
        labels = []
        keys = []
        
        # Iterate over all keys in the JSON data
        for key in data.keys():
            print(key)
            for item in data[key][:10]:  # Extract up to 10 frames per key
                landmarks = item.get('landmarks', [])
                if landmarks:  # Check if 'landmarks' is present and not empty
                    feature_row = extract_features(landmarks)
                    features.append(feature_row)
                    labels.append(item.get('slouch'))#want to fail if label not found will handle better later
                    keys.append(key)
        
        # Create DataFrame
        if features:
            num_landmarks = len(data[next(iter(data))][0]['landmarks'])  # Use the first key to get num_landmarks

            columns = [f'{feature}{i}' for i in range(num_landmarks) for feature in ['x', 'y', 'z', 'visibility']]
            print(columns)
            df = pd.DataFrame(features, columns=columns)
            df['slouch'] = labels
            df['key'] = keys  # Add a column for the key name
        else:
            df = pd.DataFrame(columns=['key', 'slouch'])  # Empty DataFrame with columns only

        return df


if __name__ == "__main__":
    # Path to your JSON file
    file_path = 'data/posture_data.json'
    df = process_json(file_path)
    #print(df)
    # Separate features and labels
    X = df.drop(['slouch','key'],axis=1)
    y = df['slouch']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    # Initialize and train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)
    joblib.dump(model, 'data/model.pkl')

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")
    with open(file_path, 'r') as file:
        new_data = json.load(file)
        new_landmarks = new_data['incorrect_hands'][0]['landmarks']
        new_feature_row = [extract_features(new_landmarks)]
        # Predict
        print(new_feature_row)
        new_predictions = model.predict(new_feature_row)
        print(f"Prediction: {new_predictions[0]}")
        print(f"actual: {new_data['incorrect_hands'][0]['slouch']}")
    