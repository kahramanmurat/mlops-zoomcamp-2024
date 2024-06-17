FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

WORKDIR /app

# Install pipenv
RUN pip install pipenv

# Copy Pipfile and Pipfile.lock
COPY Pipfile Pipfile.lock ./

# Install dependencies using pipenv
RUN pipenv install --deploy --ignore-pipfile

# Copy the script into the container
COPY starter.py .

# Set the entrypoint
ENTRYPOINT ["pipenv", "run", "python", "starter.py"]
