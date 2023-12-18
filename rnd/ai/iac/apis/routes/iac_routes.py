"""Analyze my google calendar using chatgpt in a native way

# Functionalities:
- It uses gpt-3.5
- It has a dockerized, fastapi-based chat endpoint. It uses Swagger UI
- /native_chat/ reads a calendar w/ start & end date. It can answer basic questions, e.g. who do I meet the most?
- /save_calendar_summaries/ can save one user's entire calendar 
- With /native_chat/ and /read_calendar_summaries/, one can achieve RAG: retrieval-augmented generation

# Additional Functionalities:
- It can select summary fields so that one can put more data in it
- Rate limit issue is mitigated w/ exponential backoff

# 
"""
from fastapi import APIRouter
import json
from pydantic import BaseModel
from typing import List

from rnd.ai.iac.serving import bee_qa, save_pdfs


router_prefix = f"/ai/iac"

router = APIRouter(prefix=router_prefix, tags=["iac"])


class QAInputParameters(BaseModel):
    gpt_model: str = "gpt-3.5-turbo-16k"

    prompt_template: str = "My bee questions are stored inside a sqlite3 table: qa. It has the following columns and short descriptions: (1) 'qustion_no' is the question number; (2) 'question' is the question; (3) 'answer' is the answer. When I make a request below, I want you to write the sql query and also run the sql and get me the final output."

    prompt: str = "Now can you select 3 random questions and answers?"

    verbose: int = 1


@router.post("/bee_qa")
async def beeqa(in_params: QAInputParameters):
    prompt = f"{in_params.prompt_template}. {in_params.prompt}"
    result = bee_qa.main(gpt_model=in_params.gpt_model, prompt=prompt, verbose=in_params.verbose)
    return result


########################################


class SaveInputParameters(BaseModel):
    folder_id: str = "1jB8RaTbwdr09gUukcq-orS4NsFHR-9f2"
    write_mode: str = "replace"
    verbose: int = 1


@router.post("/save_pdfs")
async def savepdfs(in_params: SaveInputParameters):
    _ = save_pdfs.main(
        folder_id=in_params.folder_id,
        write_mode=in_params.write_mode,
        verbose=in_params.verbose,
    )
    return {"status": "success"}
