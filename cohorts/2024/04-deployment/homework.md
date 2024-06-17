## Homework

In this homework, we'll deploy the ride duration model in batch mode. Like in homework 1, we'll use the Yellow Taxi Trip Records dataset. 

You'll find the starter code in the [homework](homework) directory.


## Q1. Notebook

We'll start with the same notebook we ended up with in homework 1.
We cleaned it a little bit and kept only the scoring part. You can find the initial notebook [here](homework/starter.ipynb).

Run this notebook for the March 2023 data.

What's the standard deviation of the predicted duration for this dataset?

* 1.24
* 6.24
* 12.28
* 18.28

>Answer: 6.247

Code:
```python
std_pred_duration = y_pred.std()
print(f'Standard deviation of predicted duration: {std_pred_duration}')
```


## Q2. Preparing the output

Like in the course videos, we want to prepare the dataframe with the output. 

First, let's create an artificial `ride_id` column:

```python
df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
```

Next, write the ride id and the predictions to a dataframe with results. 

Save it as parquet:

```python
df_result.to_parquet(
    output_file,
    engine='pyarrow',
    compression=None,
    index=False
)
```

What's the size of the output file?

* 36M
* 46M
* 56M
* 66M

__Note:__ Make sure you use the snippet above for saving the file. It should contain only these two columns. For this question, don't change the
dtypes of the columns and use `pyarrow`, not `fastparquet`. 

> Answer: 68M ~ 66M

## Q3. Creating the scoring script

Now let's turn the notebook into a script. 

Which command you need to execute for that?

> Answer: 

```bash
jupyter nbconvert --to script starter.ipynb
``` 


## Q4. Virtual environment

Now let's put everything into a virtual environment. We'll use pipenv for that.

Install all the required libraries. Pay attention to the Scikit-Learn version: it should be the same as in the starter
notebook.

After installing the libraries, pipenv creates two files: `Pipfile`
and `Pipfile.lock`. The `Pipfile.lock` file keeps the hashes of the
dependencies we use for the virtual env.

What's the first hash for the Scikit-Learn dependency?

```json
"scikit-learn": {
            "hashes": [
                "sha256:08ef968f6b72033c16c479c966bf37ccd49b06ea91b765e1cc27afefe723920b",
                "sha256:158faf30684c92a78e12da19c73feff9641a928a8024b4fa5ec11d583f3d8a87",
                "sha256:16455ace947d8d9e5391435c2977178d0ff03a261571e67f627c8fee0f9d431a",
                "sha256:245c9b5a67445f6f044411e16a93a554edc1efdcce94d3fc0bc6a4b9ac30b752",
```


## Q5. Parametrize the script

Let's now make the script configurable via CLI. We'll create two 
parameters: year and month.

Run the script for April 2023. 

What's the mean predicted duration? 

* 7.29
* 14.29
* 21.29
* 28.29

Hint: just add a print statement to your script.

>Answer: 14.29

Code:

```python
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
```


## Q6. Docker container 

Finally, we'll package the script in the docker container. 
For that, you'll need to use a base image that we prepared. 

This is what the content of this image is:
```
FROM python:3.10.13-slim

WORKDIR /app
COPY [ "model2.bin", "model.bin" ]
```

Note: you don't need to run it. We have already done it.

It is pushed it to [`agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim`](https://hub.docker.com/layers/agrigorev/zoomcamp-model/mlops-2024-3.10.13-slim/images/sha256-f54535b73a8c3ef91967d5588de57d4e251b22addcbbfb6e71304a91c1c7027f?context=repo),
which you need to use as your base image.

That is, your Dockerfile should start with:

```docker
FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

# do stuff here
```

This image already has a pickle file with a dictionary vectorizer
and a model. You will need to use them.

Important: don't copy the model to the docker image. You will need
to use the pickle file already in the image. 

Now run the script with docker. What's the mean predicted duration
for May 2023? 

* 0.19
* 7.24
* 14.24
* 21.19


## Bonus: upload the result to the cloud (Not graded)

Just printing the mean duration inside the docker image 
doesn't seem very practical. Typically, after creating the output 
file, we upload it to the cloud storage.

Modify your code to upload the parquet file to S3/GCS/etc.


## Bonus: Use Mage for batch inference

Here we didn't use any orchestration. In practice we usually do.

* Split the code into logical code blocks
* Use Mage to orchestrate the execution

## Publishing the image to dockerhub

This is how we published the image to Docker hub:

```bash
docker build -t mlops-zoomcamp-model:2024-3.10.13-slim .
docker tag mlops-zoomcamp-model:2024-3.10.13-slim agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

docker login --username USERNAME
docker push agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim
```

This is just for your reference, you don't need to do it.


## Submit the results

* Submit your results here: https://courses.datatalks.club/mlops-zoomcamp-2024/homework/hw4
* It's possible that your answers won't match exactly. If it's the case, select the closest one.
