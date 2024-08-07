import pandas as pd
import json
import numpy as np


# Function to extract features from landmarks
def extract_features(landmarks):
    feature_list = []
    for landmark in landmarks:
        feature_list.extend([landmark['x'], landmark['y'], landmark['z'], landmark['visibility']])
    return feature_list

def process_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
        features = []
        labels = []
        
        for item in data['incorrect_hands'][:10]:  # Extract up to 10 frames
            landmarks = item['landmarks']
            feature_row = extract_features(landmarks)
            features.append(feature_row)
            labels.append(item['slouch'])

        # Create DataFrame
        num_landmarks = len(data['incorrect_hands'][0]['landmarks'])
        num_features = num_landmarks * 4  # Each landmark has 4 features
        columns = [f'{feature}_{i}' for feature in ['x', 'y', 'z', 'visibility'] for i in range(num_landmarks)]
        df = pd.DataFrame(features, columns=columns)
        df['slouch'] = labels
        return df


if __name__ == "__main__":
    # Path to your JSON file
    file_path = 'data/posture_data.json'
    df = process_json(file_path)
    print(df)

    