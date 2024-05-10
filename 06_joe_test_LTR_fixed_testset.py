import pandas as pd
import xgboost as xgb
import numpy as np

# Load the data
def load_data(filename):
    return pd.read_csv(filename)

# Calculate the relevance scores for training data
def calculate_relevance(df):
    book_weight = 10
    click_weight = 5
    total_positions = df['position'].max()
    # Ensure the relevance score is an integer
    df['relevance_score'] = np.round((df['booking_bool'] * book_weight) + (df['click_bool'] * click_weight) - (df['position'] / total_positions * 4)).astype(int)
    return df


# Prepare the DMatrix for XGBoost
def prepare_dmatrix(features, labels=None, group_data=None):
    if labels is not None:
        dmatrix = xgb.DMatrix(data=features, label=labels)
        if group_data is not None:
            dmatrix.set_group(group_data)
    else:
        dmatrix = xgb.DMatrix(data=features)
    return dmatrix

# Train the XGBoost model
def train_model(dtrain, params):
    model = xgb.train(params, dtrain, num_boost_round=100)
    return model

# Predict using the XGBoost model
def predict(model, dtest):
    predictions = model.predict(dtest)
    return predictions

# Main function to run the program
def main():
    train_df = load_data('dmt-2024-2nd-assignment/training_set_VU_DM.csv')
    test_df = load_data('dmt-2024-2nd-assignment/test_set_VU_DM.csv')
    print("both loaded")
    
    train_df = calculate_relevance(train_df)
    

    # Prepare data for training
    train_dmatrix = prepare_dmatrix(train_df.drop(['srch_id', 'prop_id', 'click_bool', 'booking_bool', 'position', 'relevance_score', 'date_time', 'gross_bookings_usd'], axis=1),
                                    train_df['relevance_score'], train_df.groupby('srch_id').size().to_numpy())
    
    # Prepare data for testing
    test_dmatrix = prepare_dmatrix(test_df.drop(['srch_id', 'prop_id', 'date_time'], axis=1))
    print("matrixes")

    params = {
        'objective': 'rank:pairwise',
        'eta': 0.1,
        'gamma': 1.0,
        'min_child_weight': 0.1,
        'max_depth': 6,
        'eval_metric': 'ndcg@5'
    }

    
    model = train_model(train_dmatrix, params)
    predictions = predict(model, test_dmatrix)

    test_df['predicted_relevance'] = predictions
    ordered_results = test_df.sort_values(by=['srch_id', 'predicted_relevance'], ascending=[True, False])
    final_output = ordered_results[['srch_id', 'prop_id']]

    # Save the final output to a CSV file
    final_output.to_csv('sorted_properties_by_relevance.csv', index=False)
    print("Model training and prediction complete. Results saved to sorted_properties_by_relevance.csv.")
    return final_output

# Execute the program
if __name__ == "__main__":
    final_output = main()
    print(final_output.head())
