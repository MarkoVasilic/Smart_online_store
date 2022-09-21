from typing import Optional
from fastapi import FastAPI, File, UploadFile, status, HTTPException
import datetime
import pandas as pd
import sys
import time
import aiofiles
import classification_algorithms
import cluster_algorithms
import k_means_from_scratch
import distilbert_sentiment
import preprocess_words
import bart_summarization
import roberta_sentiment
from pydantic import BaseModel
import uvicorn
sys.path.insert(0, '../src')

app = FastAPI(title="Data Science Internship Project")

class UploadedFile(BaseModel):
    file_name: str

@app.post("/upload", response_model=UploadedFile)
async def upload(file: UploadFile = File(...)):
    if file.content_type != "text/csv":
        raise HTTPException(400, detail="Invalid document type, only csv files can be uploaded")
    else:
        current_time = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H-%M-%S')
        file_name = "../data/" + str(current_time) + ".csv"
        print(file.content_type)
        async with aiofiles.open(file_name, 'wb') as out_file:
            content = await file.read()  # async read
            await out_file.write(content)  # async write
        uploaded_file = UploadedFile(file_name=str(current_time))
        return uploaded_file

class ClassificationResult(BaseModel):
    documents_classified_correctly: int
    documents_classified_incorrectly: int
    classifier_score: float
    classification_report: str

@app.post("/classification", response_model=ClassificationResult)
def classification_algorithm(*, selection: str, num_of_rows: Optional[int] = 100, file_name: str):
    file_path = "../data/" + file_name + ".csv"
    res = ""
    try:
        if num_of_rows < 1:
            raise HTTPException(status_code = status.HTTP_400_BAD_REQUEST, detail="Wrong number of rows, rows must be greater than 1")
        elif selection == "MNB":
            res = classification_algorithms.multinomial_NB(num_of_rows, file_path)
        elif selection == "RF":
            res = classification_algorithms.random_forest(num_of_rows, file_path)
        elif selection == "LR":
            res = classification_algorithms.logistic_regresion(num_of_rows, file_path)
        elif selection == "SVM":
            res = classification_algorithms.support_vector_machines(num_of_rows, file_path)
        else:
            raise HTTPException(status_code = status.HTTP_400_BAD_REQUEST, detail="Wrong selection type, avaliable types (MNB, RF, LR, SVM)")
        return_value = ClassificationResult(documents_classified_correctly=res['documents_classified_correctly'], documents_classified_incorrectly=res['documents_classified_incorrectly'],
        classifier_score=res['classifier_score'], classification_report=res['classification_report'])
        return return_value
    except Exception as err:
        if isinstance(err, HTTPException):
            raise err
        raise HTTPException(status_code = status.HTTP_404_NOT_FOUND, detail="File not found")

class ClusterResult(BaseModel):
    num_of_clusters: int
    average_silhouette_score: float
    calinski_harabasz_index: float
    davies_bouldin_index: float
    homogeneity_score: float

@app.post("/cluster", response_model=ClusterResult)
def cluster_algorithm(*, selection: str, num_of_rows: Optional[int] = 100, num_of_max_clusters: Optional[int] = 5, file_name: str):
    file_path = "../data/" + file_name + ".csv"
    res = ""
    try:
        if num_of_rows < 1 or num_of_max_clusters < 2 or num_of_max_clusters > 20:
            raise HTTPException(status_code = status.HTTP_400_BAD_REQUEST, detail="Wrong number of rows or number of clusters, rows must be greater than 1, and number of clusters between 2 and 20")
        elif selection == "KM5":
            res = cluster_algorithms.k_means_5_clusters(num_of_rows, file_path)
        elif selection == "KM":
            res = cluster_algorithms.evaluate_k_means(num_of_rows, num_of_max_clusters, file_path)
        elif selection == "DBS":
            res = cluster_algorithms.dbscan(num_of_rows, file_path)
        else:
            raise HTTPException(status_code = status.HTTP_400_BAD_REQUEST, detail="Wrong selection type, avaliable types (KM5, KM, DBS)")
        return_value = ClusterResult(num_of_clusters=res['Num_of_clusters'], average_silhouette_score=res['Average_silhouette_score'], calinski_harabasz_index=res['Calinski_Harabasz_Index'],
        davies_bouldin_index=res['Davies_Bouldin_Index'], homogeneity_score=res['Homogeneity_score'])
        return return_value
    except Exception as err:
        if isinstance(err, HTTPException):
            raise err
        raise HTTPException(status_code = status.HTTP_404_NOT_FOUND, detail="File not found")

class FromScratchResult(BaseModel):
    num_of_clusters: int
    average_silhouette_score: float

@app.post("/kmeans_from_scratch", response_model=FromScratchResult)
def kmeans_from_scratch_algorithm(*, num_of_rows: Optional[int] = 100, num_of_max_clusters: Optional[int] = 5, file_name: str):
    file_path = "../data/" + file_name + ".csv"
    res = ""
    try:
        if num_of_rows < 1 or num_of_max_clusters < 2 or num_of_max_clusters > 20:
            raise HTTPException(status_code = status.HTTP_400_BAD_REQUEST, detail="Wrong number of rows or number of clusters, rows must be greater than 1, and number of clusters between 2 and 20")
        else:
            res = k_means_from_scratch.k_means_from_scratch(num_of_rows, num_of_max_clusters, file_path)
        return_value = FromScratchResult(num_of_clusters=res["Num_of_clusters"], average_silhouette_score=res['Average_silhouette_score'])
        return return_value
    except Exception as err:
        if isinstance(err, HTTPException):
            raise err
        raise HTTPException(status_code = status.HTTP_404_NOT_FOUND, detail="File not found")

class InputText(BaseModel):
    text: str

class SentimentText(BaseModel):
    sentiment: str

class SummaryText(BaseModel):
    summary: str

@app.post("/distilbert_sentiment", response_model=SentimentText)
def distilbert_sentiment_analysis(input_text: InputText):
    return_value = SentimentText(sentiment=distilbert_sentiment.do_sentiment(preprocess_words.text_preprocessing_eng(pd.DataFrame([input_text.text], columns=['text']), 'text')['text'][0]))
    return return_value

@app.post("/roberta_sentiment", response_model=SentimentText)
def roberta_sentiment_analysis(input_text: InputText):
    return_value = SentimentText(sentiment=roberta_sentiment.do_sentiment(preprocess_words.text_preprocessing_eng(pd.DataFrame([input_text.text], columns=['text']), 'text')['text'][0]))
    return return_value

@app.post("/summarization", response_model=SummaryText)
def summarization(input_text: InputText):
    return_value = SummaryText(summary=bart_summarization.do_summarization(input_text.text))
    return return_value

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port = 8000)