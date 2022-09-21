FROM python:3.9.12
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE $PORT
CMD uvicorn fast_api:app --reload --port=$PORT