import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report



df = pd.read_csv('feature_eng_out.csv')

features = [
    'prop_starrating', 'prop_review_score', 'prop_location_score1', 
    'prop_location_score2', 'prop_log_historical_price', 'price_usd', 
    'promotion_flag', 'srch_length_of_stay', 'srch_booking_window', 
    'srch_adults_count', 'srch_children_count', 'srch_room_count', 
    'orig_destination_distance', 'srch_saturday_night_bool'
]

# Handling missing values - Simple Imputation
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(df[features])
y = df['click_bool']

# Normalize the feature data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)



X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


# Build the MLP model
#mlp = MLPClassifier(hidden_layer_sizes=(100, ), max_iter=300, activation='relu', solver='adam', random_state=1)


model = LogisticRegression()
model.fit(X_train, y_train)

# Predict probabilities
probabilities = model.predict_proba(df[features])[:, 1]  # Get the probability of '1' (click)

# Add probabilities back to the original DataFrame
df['predicted_probability'] = probabilities

# Sort by 'srch_id' and 'predicted_probability'
sorted_df = df.sort_values(by=['srch_id', 'predicted_probability'], ascending=[True, False])

# Select only the necessary columns to output
output_df = sorted_df[['srch_id', 'prop_id']]

# To display the DataFrame
print(output_df.head())

# Or export to CSV
output_df.to_csv('sorted_click_predictions.csv', index=False)


