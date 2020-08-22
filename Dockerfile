FROM python:3.6-slim-stretch

RUN apt-get update && apt-get install -y git python3-dev gcc \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
RUN apt-get update -y && apt-get install -y --no-install-recommends build-essential gcc \
    libsndfile1 

COPY app/requirements.txt .
COPY app/requirements-fastai2audio.txt .
RUN pip install --upgrade -r requirements.txt

COPY app app/

EXPOSE 5000

CMD ["python", "app/server.py", "serve"]
