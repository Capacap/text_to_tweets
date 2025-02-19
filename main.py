import os
import uvicorn

from mistralai import Mistral
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel, Field
from tweets_from_text import tweets_from_text

app = FastAPI(title="TEXT TO TWEETS API")
model = os.environ["MISTRAL_MODEL"]
client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

TEXT_MAX_LENGTH = 20000
TEXT_MIN_LENGTH = 10
DEFAULT_CHUNK_SIZE = 512
DEFAULT_OVERLAP_SIZE = 64

class ProcessRequest(BaseModel):
    text: str = Field(..., max_length=TEXT_MAX_LENGTH, description="The text to be translated into tweets.")
    chunk_size: int = Field(default=DEFAULT_CHUNK_SIZE, gt=0, le=1024, description="Must be between 1-1024 characters.")
    overlap_size: int = Field(default=DEFAULT_OVERLAP_SIZE, gt=0, le=1024, description="Must be between 1-1024 characters.")

class TweetListResponse(BaseModel):
    tweets: list[str]

@app.post("/process/", response_model=TweetListResponse)
async def process_text(request: ProcessRequest):
    try:
        cleaned_text = request.text.strip()
        if not cleaned_text:
            raise HTTPException(status_code=400, detail="No text provided.")
        if len(cleaned_text) < TEXT_MIN_LENGTH:
            raise HTTPException(status_code=400, detail="Text too short.")
        if len(cleaned_text) > TEXT_MAX_LENGTH:
            raise HTTPException(status_code=400, detail="Text exceeds maximum length.")

        result = tweets_from_text(
            cleaned_text, 
            request.chunk_size,
            request.overlap_size,
            client, 
            model
        )
        
        return {"tweets": result}
        
    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Processing error: {str(e)}"
        )

@app.post("/process-plaintext/", response_model=TweetListResponse)
async def process_plaintext(text: str = Body(..., media_type='text/plain', max_length=TEXT_MAX_LENGTH, description="The text to be translated into tweets.")):
    try:
        cleaned_text = text.strip()
        if len(cleaned_text) < TEXT_MIN_LENGTH:
            raise HTTPException(status_code=400, detail="Text too short.")
        if len(cleaned_text) > TEXT_MAX_LENGTH:
            raise HTTPException(status_code=400, detail="Text exceeds maximum length.")

        result = tweets_from_text(
            cleaned_text, 
            DEFAULT_CHUNK_SIZE,
            DEFAULT_OVERLAP_SIZE,
            client, 
            model
        )
        
        return {"tweets": result}
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Processing error: {str(e)}"
        )

if __name__ == "__main__":
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True
    )
