from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from .rag_chain import answer_question

app = FastAPI()

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str

@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    try:
        answer = answer_question(request.question)
        return QueryResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))