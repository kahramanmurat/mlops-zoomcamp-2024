
## Question 1. Run Mage

First, let's run Mage with Docker Compose. Follow the quick start guideline. 

What's the version of Mage we run? 

(You can see it in the UI)

Answer:
```v0.9.70```

## Question 2. Creating a project

Now let's create a new project. We can call it "homework_03", for example.

How many lines are in the created `metadata.yaml` file? 

- 35
- 45
- 55 - correct
- 65

Answer:
```55 lines```

## Question 3. Creating a pipeline

Let's create an ingestion code block.

In this block, we will read the March 2023 Yellow taxi trips data.

How many records did we load? 

- 3,003,766
- 3,203,766
- 3,403,766 - correct
- 3,603,766

Answer:
```3403766```

Code:

```
import requests
from io import BytesIO
from typing import List

import pandas as pd

if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader


@data_loader
def ingest_files(**kwargs) -> pd.DataFrame:
    dfs: List[pd.DataFrame] = []

    for year, months in [(2023, (3, 4))]:  # Update range to ensure it includes March
        for i in range(*months):
            print(f"Fetching data for {year}-{i:02d}")
            response = requests.get(
                f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{i:02d}.parquet'
            )

            if response.status_code != 200:
                raise Exception(f"Failed to fetch data for {year}-{i:02d}: {response.text}")

            df = pd.read_parquet(BytesIO(response.content))
            print(f"Data shape for {year}-{i:02d}: {df.shape}")
            dfs.append(df)

    if not dfs:
        raise ValueError("No dataframes to concatenate")

    result_df = pd.concat(dfs)
    print(f"Total records loaded: {result_df.shape[0]}")
    return result_df
```


## Question 4. Data preparation


Let's use the same logic for preparing the data we used previously. We will need to create a transformer code block and put this code there.

This is what we used (adjusted for yellow dataset):

```python
def read_dataframe(filename):
    df = pd.read_parquet(filename)

    df.tpep_dropoff_datetime = pd.to_datetime(df.tpep_dropoff_datetime)
    df.tpep_pickup_datetime = pd.to_datetime(df.tpep_pickup_datetime)

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    
    return df
```

Let's adjust it and apply to the data we loaded in question 3. 

What's the size of the result? 


- 2,903,766
- 3,103,766
- 3,316,216 - correct
- 3,503,766


Answer:
```3316216```

Code:
```
import pandas as pd

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@transformer
def transform(data, *args, **kwargs):
    """
    Template code for a transformer block.

    Add more parameters to this function if this block has multiple parent blocks.
    There should be one parameter for each output variable from each parent block.

    Args:
        data: The output from the upstream parent block
        args: The output from any additional upstream blocks (if applicable)

    Returns:
        DataFrame: Transformed data frame
    """
    # Specify your transformation logic here
    df = data.copy()

    df.tpep_dropoff_datetime = pd.to_datetime(df.tpep_dropoff_datetime)
    df.tpep_pickup_datetime = pd.to_datetime(df.tpep_pickup_datetime)

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    
    return df


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
    assert isinstance(output, pd.DataFrame), 'The output is not a DataFrame'
    assert 'duration' in output.columns, 'The output DataFrame does not contain the "duration" column'
    assert output.shape[0] > 0, 'The output DataFrame is empty'
```

## Question 5. Train a model

We will now train a linear regression model using the same code as in homework 1

* Fit a dict vectorizer
* Train a linear regression with default parameres 
* Use pick up and drop off locations separately, don't create a combination feature

Let's now use it in the pipeline. We will need to create another transformation block, and return both the dict vectorizer and the model

What's the intercept of the model? 

Hint: print the `intercept_` field in the code block

- 21.77
- 24.77 - correct
- 27.77
- 31.77

Answer:
```24.77```

Code:
```
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

# Model Training Function
@transformer
def train_model(data, *args, **kwargs):
    """
    Train a linear regression model using the transformed data.

    Args:
        data: The transformed data from the previous block.

    Returns:
        tuple: A tuple containing the DictVectorizer and the trained Linear Regression model.
    """

    # Extract relevant columns for training
    features = data[['PULocationID', 'DOLocationID']]
    target = data['duration']

    # Convert the features to a dictionary format suitable for DictVectorizer
    features_dict = features.to_dict(orient='records')

    # Initialize and fit the DictVectorizer
    dv = DictVectorizer()
    X = dv.fit_transform(features_dict)

    # Initialize and train the Linear Regression model
    model = LinearRegression()
    model.fit(X, target)

    # Print the intercept of the model
    print(f"Model intercept: {model.intercept_}")

    return dv, model
```

## Question 6. Register the model 

The model is trained, so let's save it with MLFlow.

If you run mage with docker-compose, stop it with Ctrl+C or 

```bash
docker-compose down
```

Let's create a dockerfile for mlflow, e.g. `mlflow.dockerfile`:

```dockerfile
FROM python:3.10-slim

RUN pip install mlflow==2.12.1

EXPOSE 5000

CMD [ \
    "mlflow", "server", \
    "--backend-store-uri", "sqlite:///home/mlflow/mlflow.db", \
    "--host", "0.0.0.0", \
    "--port", "5000" \
]
```

And add it to the docker-compose.yaml:

```yaml
  mlflow:
    build:
      context: .
      dockerfile: mlflow.dockerfile
    ports:
      - "5000:5000"
    volumes:
      - "${PWD}/mlflow:/home/mlflow/"
    networks:
      - app-network
```

Note that `app-network` is the same network as for mage and postgre containers.
If you use a different compose file, adjust it.

We should already have `mlflow==2.12.1` in requirements.txt in the mage project we created for the module. If you're starting from scratch, add it to your requirements.

Next, start the compose again and create a data exporter block.

In the block, we

* Log the model (linear regression)
* Save and log the artifact (dict vectorizer)

If you used the suggested docker-compose snippet, mlflow should be accessible at `http://mlflow:5000`.

Find the logged model, and find MLModel file. What's the size of the model? (`model_size_bytes` field):

* 14,534
* 9,534
* 4,534 - correct
* 1,534

> Note: typically we do two last steps in one code block 

Answer:
```4,534```

Code:
```import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
import mlflow
import mlflow.sklearn
import joblib
import os

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter

@data_exporter
def export_data(data, *args, **kwargs):
    """
    Exports data to some source.

    Args:
        data: The output from the upstream parent block
        args: The output from any additional upstream blocks (if applicable)

    Output (optional):
        Optionally return any object and it'll be logged and
        displayed when inspecting the block run.
    """
    # Specify your data exporting logic here
    dv, model = data
    
    # Set the experiment name
    experiment_name = "my_new_experiment"  # Change this to your desired experiment name
    mlflow.set_experiment(experiment_name)
    
    # Ensure the experiment is created and get its ID
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id
    
    # Set the tracking URI to point to the running MLflow server
    mlflow.set_tracking_uri("http://mlflow:5000")
    
    with mlflow.start_run(experiment_id=experiment_id):
        mlflow.sklearn.log_model(model, "linear_regression_model")
        mlflow.log_param("intercept", model.intercept_)
        
        # Save the DictVectorizer
        dv_path = "dict_vectorizer.pkl"
        joblib.dump(dv, dv_path)
        mlflow.log_artifact(dv_path)

    return data```


## Submit the results

* Submit your results here: https://courses.datatalks.club/mlops-zoomcamp-2024/homework/hw3
* If your answer doesn't match options exactly, select the closest one


