#!/usr/bin/env python
# coding: utf-8

import pickle
import pandas as pd
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='Predict taxi trip durations.')
parser.add_argument('--year', type=int, required=True, help='Year of the data')
parser.add_argument('--month', type=int, required=True, help='Month of the data')
args = parser.parse_args()

year = args.year
month = args.month

with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)

categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df

# Read data for the specified year and month
data_file = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
df = read_data(data_file)

# Create ride_id column
df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

# Prepare the data for prediction
dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = model.predict(X_val)

# Print the mean predicted duration
mean_pred_duration = y_pred.mean()
print(f'Mean predicted duration: {mean_pred_duration}')

# Create result dataframe
df_result = pd.DataFrame({'ride_id': df['ride_id'], 'predicted_duration': y_pred})

# Save result dataframe as parquet file
output_file = f'result_{year:04d}-{month:02d}.parquet'
df_result.to_parquet(output_file, engine='pyarrow', compression=None, index=False)