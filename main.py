from typing import Union
from fastapi import FastAPI

from face_detection import detect_face

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/face")
def face():

    return detect_face()