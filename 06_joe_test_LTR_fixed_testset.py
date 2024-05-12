import pandas as pd
import xgboost as xgb
import numpy as np

# Load the data
def load_data(filename):
    return pd.read_csv(filename)

# Calculate the relevance scores for training data
def calculate_relevance(df):
    book_weight = 5
    click_weight = 2
    total_positions = df['position'].max()
    # Ensure the relevance score is an integer
    df['relevance_score'] = np.round((df['booking_bool'] * book_weight) + (df['click_bool'] * click_weight) +total_positions/df['position']/4).astype(int)
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
def get_time_of_day(hour):
    if 5 <= hour < 12:
        return 1
    elif 12 <= hour < 17:
        return 2
    elif 17 <= hour < 21:
        return 3
    else:
        return 4

# Main function to run the program
def main():
    train_df = load_data('training_set_VU_DM.csv')

    train_df['date_time'] = pd.to_datetime(train_df['date_time'])
    train_df['time_of_day'] = train_df['date_time'].dt.hour.apply(get_time_of_day)
    train_df['is_weekend'] = train_df['date_time'].dt.dayofweek >= 5
    train_df['price_per_night'] = train_df['price_usd'] / train_df['srch_length_of_stay']
    train_df['star_rating_difference'] = train_df['prop_starrating'] - train_df['visitor_hist_starrating'].fillna(train_df['prop_starrating'])
    train_df['is_domestic'] = (train_df['prop_country_id'] == train_df['visitor_location_country_id']).astype(int)
    train_df[[f'comp{i}_rate' for i in range(1, 9)]].fillna(0, inplace=True)

    train_df['rate_advantage_count'] = (train_df[[f'comp{i}_rate' for i in range(1, 3)]].fillna(0) > 0).sum(axis=1)
    train_df['rate_disadvantage_count'] = (train_df[[f'comp{i}_rate' for i in range(1, 3)]].fillna(0) < 0).sum(axis=1)
    train_df['availability_advantage_count'] = (train_df[[f'comp{i}_inv' for i in range(1, 3)]].fillna(0) > 0).sum(axis=1)
    train_df['avg_rate_percent_diff'] = train_df[[f'comp{i}_rate_percent_diff' for i in range(1, 3)]].mean(axis=1, skipna=True)
    train_df['max_rate_percent_diff'] = train_df[[f'comp{i}_rate_percent_diff' for i in range(1, 3)]].max(axis=1, skipna=True)
    train_df['min_rate_percent_diff'] = train_df[[f'comp{i}_rate_percent_diff' for i in range(1, 3)]].min(axis=1, skipna=True)

    test_df = load_data('test_set_VU_DM.csv')


    test_df['date_time'] = pd.to_datetime(test_df['date_time'])
    test_df['time_of_day'] = test_df['date_time'].dt.hour.apply(get_time_of_day)
    test_df['is_weekend'] = test_df['date_time'].dt.dayofweek >= 5
    test_df['price_per_night'] = test_df['price_usd'] / test_df['srch_length_of_stay']
    test_df['star_rating_difference'] = test_df['prop_starrating'] - test_df['visitor_hist_starrating'].fillna(test_df['prop_starrating'])
    test_df['is_domestic'] = (test_df['prop_country_id'] == test_df['visitor_location_country_id']).astype(int)
    test_df[[f'comp{i}_rate' for i in range(1, 9)]].fillna(0, inplace=True)

    test_df['rate_advantage_count'] = (test_df[[f'comp{i}_rate' for i in range(1, 3)]].fillna(0) > 0).sum(axis=1)
    test_df['rate_disadvantage_count'] = (test_df[[f'comp{i}_rate' for i in range(1, 3)]].fillna(0) < 0).sum(axis=1)
    test_df['availability_advantage_count'] = (test_df[[f'comp{i}_inv' for i in range(1, 3)]].fillna(0) > 0).sum(axis=1)
    test_df['avg_rate_percent_diff'] = test_df[[f'comp{i}_rate_percent_diff' for i in range(1, 3)]].mean(axis=1, skipna=True)
    test_df['max_rate_percent_diff'] = test_df[[f'comp{i}_rate_percent_diff' for i in range(1, 3)]].max(axis=1, skipna=True)
    test_df['min_rate_percent_diff'] = test_df[[f'comp{i}_rate_percent_diff' for i in range(1, 3)]].min(axis=1, skipna=True)
    
    
    
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
