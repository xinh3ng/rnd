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

from xhcaftv.commons.commons import create_logger
from xhcaftv.ai.calendar.serving import calendar_chat_native, calendar_chat_rag, save_calendar_summaries


logger = create_logger(__name__)


########################################


router_prefix = f"/ai/calendar"

router = APIRouter(prefix=router_prefix, tags=["in_app"])


class NativeChatInputParameters(BaseModel):
    gpt_model: str = "gpt-3.5-turbo-16k"
    prompt_template: str = (
        "This is my calendar: {calendar_summaries}. List top-10 attendees that I meet the most frequent?"
    )
    date_range: List[str] = ["2023-10-10", "2023-10-31"]
    calendar_fields: List[str] = ["title", "description", "start", "end", "organizer", "attendees"]
    verbose: int = 1


@router.post("/native_chat")
async def native_chat(in_params: NativeChatInputParameters):
    result = calendar_chat_native.main(
        gpt_model=in_params.gpt_model,
        prompt_template=in_params.prompt_template,
        date_range=in_params.date_range,
        calendar_fields=in_params.calendar_fields,
        session_id=None,
        verbose=in_params.verbose,
    )
    return result


class RagChatInputParameters(BaseModel):
    gpt_model: str = "gpt-3.5-turbo-16k"
    prompt: str = ""
    verbose: int = 1


@router.post("/rag_chat")
async def rag_chat(in_params: RagChatInputParameters):
    result = calendar_chat_rag.main(gpt_model=in_params.gpt_model, prompt=in_params.prompt)
    return result


########################################


class SaveInputParameters(BaseModel):
    date_range: List[str] = ["2023-10-30", "2023-10-31"]
    calendar_fields: List[str] = None
    write_mode: str = "append"
    verbose: int = 1


@router.post("/save_calendar_summaries")
async def save_summaries(in_params: SaveInputParameters):
    _ = save_calendar_summaries.main(
        date_range=in_params.date_range,
        calendar_fields=in_params.calendar_fields,
        write_mode=in_params.write_mode,
        verbose=in_params.verbose,
    )
    return {"status": "success"}


class ReadInputParameters(BaseModel):
    sql_query: str = "select * from calendar_summary limit 5"
    verbose: int = 1


@router.post("/read_calendar_summaries")
async def read_summaries(in_params: ReadInputParameters):
    op = save_calendar_summaries.DbOperator()
    result = op.read(sql_query=in_params.sql_query)
    logger.info(f"result has {len(result)} rows")

    result = result.to_dict("records")
    if in_params.verbose >= 3:
        logger.info("result:\n%s" % json.dumps(result, indent=4))
    return result
