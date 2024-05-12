import pandas as pd
import numpy as np
import lightgbm as lgb

# Load the data
def load_data(filename):
   return pd.read_csv(filename)

# Calculate the relevance scores for training data
def calculate_relevance(df):
    df['relevance_score'] = 0

    # Set relevance_score to 1 where click_bool is 1 but not booked
    df.loc[(df['click_bool'] == 1) & (df['booking_bool'] == 0), 'relevance_score'] = 1

    # Set relevance_score to 5 where booking_bool is 1
    df.loc[df['booking_bool'] == 1, 'relevance_score'] = 5

    return df

# Prepare the Dataset for LightGBM
def prepare_lgb_data(features, labels=None, group=None):
    if labels is not None and group is not None:
        dataset = lgb.Dataset(data=features, label=labels, group=group, free_raw_data=False)
    else:
        dataset = lgb.Dataset(data=features, free_raw_data=False)
    return dataset

# Train the LightGBM model
def train_model(train_data, params):
    model = lgb.train(params, train_data, num_boost_round=100)
    return model

# Predict using the LightGBM model
def predict(model, data):
    predictions = model.predict(data)
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

def cleaning_data(df):
    df.drop(['visitor_hist_starrating', 'visitor_hist_adr_usd'], axis=1, inplace=True)
    mean_scores = df.groupby('prop_id')['prop_review_score'].mean()
    df['prop_review_score'] = df.groupby('prop_id')['prop_review_score'].transform(lambda x: x.fillna(x.mean()))




def feature_engineering(df):
    df['date_time'] = pd.to_datetime(df['date_time'])
    df['time_of_day'] = df['date_time'].dt.hour.apply(get_time_of_day)
    df['is_weekend'] = df['date_time'].dt.dayofweek >= 5
    df['price_per_night'] = df['price_usd'] / df['srch_length_of_stay']
    #df['star_rating_difference'] = df['prop_starrating'] - df['visitor_hist_starrating'].fillna(df['prop_starrating'])
    df['is_domestic'] = (df['prop_country_id'] == df['visitor_location_country_id']).astype(int)
    df['has_hist_starrating'] = df['visitor_hist_starrating'].notna().astype(int)
    df['has_hist_adr_usd'] = df['visitor_hist_adr_usd'].notna().astype(int)
    df['prop_location_overall'] = (df['prop_location_score1'] + df['prop_location_score2'])/2
    df['srch_count'] = df['srch_adults_count'] + df['srch_adults_count']

    df['date_time'] = pd.to_datetime(train_df['date_time'])
    df['time_of_day'] = df['date_time'].dt.hour.apply(get_time_of_day)
    df['is_weekend'] = df['date_time'].dt.dayofweek >= 5
    df['price_per_night'] = df['price_usd'] / df['srch_length_of_stay']
    df['star_rating_difference'] = df['prop_starrating'] - df['visitor_hist_starrating'].fillna(df['prop_starrating'])
    df['is_domestic'] = (df['prop_country_id'] == df['visitor_location_country_id']).astype(int)
    df[[f'comp{i}_rate' for i in range(1, 9)]].fillna(0, inplace=True)

    df['rate_advantage_count'] = (df[[f'comp{i}_rate' for i in range(1, 3)]].fillna(0) > 0).sum(axis=1)
    df['rate_disadvantage_count'] = (df[[f'comp{i}_rate' for i in range(1, 3)]].fillna(0) < 0).sum(axis=1)
    df['availability_advantage_count'] = (df[[f'comp{i}_inv' for i in range(1, 3)]].fillna(0) > 0).sum(axis=1)
    df['avg_rate_percent_diff'] = df[[f'comp{i}_rate_percent_diff' for i in range(1, 3)]].mean(axis=1, skipna=True)
    df['max_rate_percent_diff'] = df[[f'comp{i}_rate_percent_diff' for i in range(1, 3)]].max(axis=1, skipna=True)
    df['min_rate_percent_diff'] = df[[f'comp{i}_rate_percent_diff' for i in range(1, 3)]].min(axis=1, skipna=True)

    return df

# Main function to run the program
def main():
    train_df = load_data('training_set_VU_DM.csv')
    test_df = load_data('test_set_VU_DM.csv')

    train_df = feature_engineering(train_df)
    test_data = feature_engineering(test_df)

    train_df = calculate_relevance(train_df)

    train_data = prepare_lgb_data(train_df.drop(['srch_id', 'prop_id', 'click_bool', 'booking_bool', 'position', 'relevance_score', 'date_time', 'gross_bookings_usd'], axis=1),
                                  train_df['relevance_score'], train_df.groupby('srch_id').size().tolist())
    test_data = prepare_lgb_data(test_df.drop(['srch_id', 'prop_id', 'date_time'], axis=1))

    params = {
        'objective': 'lambdarank',
        'learning_rate': 0.1,
        'num_leaves': 70,
        'metric': 'ndcg',
        'ndcg_eval_at': [5]
    }

    model = train_model(train_data, params)
    predictions = predict(model, test_data.data)

    test_df['predicted_relevance'] = predictions
    ordered_results = test_df.sort_values(by=['srch_id', 'predicted_relevance'], ascending=[True, False])
    final_output = ordered_results[['srch_id', 'prop_id']]

    # Save the final output to a CSV file
    final_output.to_csv('sorted_properties_by_relevance.csv', index=False)
    print("Model training and prediction complete. Results saved to sorted_properties_by_relevance.csv.")
    return final_output


final_output = main()
print(final_output.head())