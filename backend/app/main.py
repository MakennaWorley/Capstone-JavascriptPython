import os
import io
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pathlib import Path

app = FastAPI()

origins = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://localhost:8501").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in origins],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
DATASETS_DIR = BASE_DIR / "datasets"

# health check
@app.get("/api/hello")
def hello():
    return {"message": "Hello from FastAPI"}

# Create Dataset
@app.post("/api/create/data")
async def create_dataset(request: Request):
    # Print a specific header (recommended)
    x_api_key = request.headers.get("x-api-key")
    print("X-API-Key:", x_api_key)

    # Print only headers you care about
    interesting = ["x-api-key", "content-type", "origin", "referer"]
    print("=== Selected Headers ===")
    for h in interesting:
        print(f"{h}: {request.headers.get(h)}")

    # Print the JSON body you sent from the form
    body = await request.json()
    print("=== Body ===")
    print(body)

    return {"message": "Received request (stub)."}