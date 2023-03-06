from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

from predict import Args, Predicter

class TextData(BaseModel):
    name: Optional[str] = None
    text: str

app = FastAPI()
args = Args()
predicter = Predicter(args)

@app.get("/")
async def root():
    return { "message": "Hello" }

@app.post("/text")
async def create_item(text: TextData):
    res = { "code": 200, "msg": "good", "data": None }
    try:
        data = predicter.predict(text.text)
        res["data"] = data
    except:
        print("error")
        res["code"] = 404
        res["error"] = "predict error"
    return res

if __name__ == '__main__':
    uvicorn.run(app)
