import os
import uvicorn

from mistralai import Mistral
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel, Field
from tweets_from_text import tweets_from_text

app = FastAPI(title="Tweet Generator API")
model = "mistral-small-latest"
client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

class ProcessRequest(BaseModel):
    text: str = Field(..., max_length=4096, description="The text to be translated into tweets.")
    chunk_size: int = Field(default=512, gt=0, le=1024, description="Must be between 1-1024 characters.")

class TweetListResponse(BaseModel):
    tweets: list[str]

@app.post("/process/", response_model=TweetListResponse)
async def process_text(request: ProcessRequest):
    try:
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="No text provided.")

        result = tweets_from_text(
            request.text.strip(), 
            request.chunk_size,
            64,
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
async def process_plaintext(text: str = Body(..., media_type='text/plain', max_length=20000)):
    try:
        cleaned_text = text.strip()
        if len(cleaned_text) < 10:
            raise HTTPException(status_code=400, detail="Text too short")
        if len(cleaned_text) > 20000:
            raise HTTPException(status_code=400, detail="Text exceeds maximum length")

        result = tweets_from_text(
            cleaned_text, 
            512,
            64,
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
