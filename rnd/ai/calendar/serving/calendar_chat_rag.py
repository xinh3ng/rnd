"""Analyze my google calendar using chatgpt in a RAG way

# Usage Example

gpt_model="gpt-3.5-turbo-16k"

prompt="My google calendar is stored inside a sqlite3 table: calendar_summary. It has the following columns and short descriptions: (1)'title' is the meeting title in string format; (2) 'description' is the meeting detailed description in string format; (3) 'start' is the meeting start time in UTC; (4) 'end' is the meeting end time in UTC; (5) 'organizer' is the meeting organizer's email; (6) 'attendees' is a json array of attendee emails, which can only be used after being exploded. Now I want to know my longest meetings and their attendees, can you write that sql?"

verbose=3

python xhcaftv/ai/calendar/serving/calendar_chat_rag.py --gpt_model=$gpt_model --prompt="$prompt" --verbose=$verbose

"""
import json
import pandas as pd
import re

from xhcaftv.commons.commons import create_logger
from xhcaftv.ai.calendar.utils.chat_utils import chat_with_backoff
from xhcaftv.ai.calendar.serving import save_calendar_summaries

# Script-level constants
logger = create_logger(__name__)

pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 120)
pd.set_option("display.max_colwidth", None)  # No truncation


def parse_sql_query(text: str) -> str:
    text = text.lower()

    # Grab the first ```sql and the immediate next ';'
    pattern = r"```.*?```"
    match = re.search(pattern, text, re.DOTALL)

    sql = text[match.start() : match.end()].replace("```", "").replace("sql", "")
    assert ("select" in sql) and ("from" in sql)
    return sql


def main(
    gpt_model: str,
    prompt: str,
    session_id: str = None,
    use_session_id: bool = False,
    verbose: int = 1,
) -> dict:
    if not use_session_id:
        session_id = None
    response = chat_with_backoff(model=gpt_model, messages=[{"role": "user", "content": prompt}], session_id=session_id)
    reply = response["choices"][0]["message"]["content"]
    logger.info("Reply: %s" % reply)

    sql_query = parse_sql_query(reply)
    op = save_calendar_summaries.DbOperator()
    result = op.read(sql_query=sql_query)

    result = {
        "session_id": response["id"],
        "reply": reply,
        "result": result.to_dict("records"),
    }
    if verbose >= 3:
        logger.info("result:\n%s" % json.dumps(result, indent=4))
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpt_model")
    parser.add_argument("--prompt")
    parser.add_argument("--verbose", type=int, default=1)
    args = vars(parser.parse_args())

    logger.info("Command line args:\n%s" % json.dumps(args, indent=4))
    main(**args)
    logger.info("ALL DONE!\n")
