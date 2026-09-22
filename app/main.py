from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .rag_chain import answer_question

app = FastAPI()

origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # ou ["*"] pour tout autoriser (déconseillé en production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    in_scope: float
    grounded: Optional[float]
    sensitivity: str
    sources: list


@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    try:
        return QueryResponse(**answer_question(request.question))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
