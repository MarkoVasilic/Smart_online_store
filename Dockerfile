FROM python:3.9.12
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
RUN python -m nltk.downloader stopwords
RUN python -m nltk.downloader punkt
WORKDIR /app/src
EXPOSE $PORT
CMD uvicorn fast_api:app --reload --port=$PORT --host='0.0.0.0'