from fastapi import FastAPI, Request
import uvicorn
import sys
import os
from fastapi.templating import Jinja2Templates
from starlette.responses import RedirectResponse
from fastapi.responses import Response
from pydantic import BaseModel
from TextSummarizer.pipeline.prediction import PredictionPipeline


text:str = "What is Text Summarization?"

app = FastAPI()
templates = Jinja2Templates(directory="templates")

class SummarizeRequest(BaseModel):
    text: str

@app.get("/", tags=["authentication"])
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})



@app.get("/train")
async def training():
    try:
        os.system("python main.py")
        return Response("Training completed successfully !!")

    except Exception as e:
        return Response(f"Error Occurred! {e}")
    



@app.post("/predict")
async def predict_route(req: SummarizeRequest):
    try:

        obj = PredictionPipeline()
        text = obj.predict(req.text)
        return text
    except Exception as e:
        raise e
    

if __name__=="__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)