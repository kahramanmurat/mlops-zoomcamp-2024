#!/usr/bin/env python
# coding: utf-8

import pickle
import pandas as pd


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

df = read_data('../data/yellow_tripdata_2023-03.parquet')

# Create ride_id column
year = 2023
month = 3
df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')


# Prepare the data for prediction
dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = model.predict(X_val)


std_pred_duration = y_pred.std()
print(f'Standard deviation of predicted duration: {std_pred_duration}')

# Create result dataframe
df_result = pd.DataFrame({'ride_id': df['ride_id'], 'predicted_duration': y_pred})

#Save result dataframe as parquet file
output_file = 'result_2023-03.parquet'
df_result.to_parquet(output_file, engine='pyarrow', compression=None, index=False)




