"""Analyze my google calendar using chatgpt in a native way

I have to feed my entire calendar into chatgpt prompt. It is easy but is limited by token length

# Functionalities:
- It takes a calendar's start and end date. It can answer basic questions, e.g. who do I meet the most?
- It uses gpt-3.5
- It has a dockerized, fastapi-based chat endpoint. It uses Swagger UI
- It can select summary fields so that one can put more data in it
- The rate limit issue is mitigated w/ exponential backoff

# TODO
- P2: cannot feed a lot of data

# gpt models and rate limits
                       TPM    RPM
gpt-3.5-turbo	    90,000	3,500
gpt-3.5-turbo-0301	90,000	3,500
gpt-3.5-turbo-0613	90,000	3,500
gpt-3.5-turbo-16k	180,000	3,500
gpt-3.5-turbo-16k-0613	180,000	3,500

# Usage Example
gpt_model=gpt-3.5-turbo-16k

prompt_template="This is my calendar: {calendar_summaries}. List top-10 attendees that I meet the most frequent?"

python rnd/ai/calendar/serving/calendar_chat_native.py --gpt_model=$gpt_model --prompt_template="$prompt_template"
"""
from datetime import datetime
from googleapiclient.discovery import build
import json
import pandas as pd
from typing import List


from rnd.ai.calendar.utils.calendar_utils import get_credentials, summarize_calendar
from rnd.ai.calendar.utils.chat_utils import chat_with_backoff


pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 120)
pd.set_option("display.max_colwidth", None)  # No truncation


def main(
    gpt_model: str,
    prompt_template: str,
    date_range: List[str] = ["2023-10-10", "2023-10-31"],
    calendar_fields: List[str] = ["title", "start", "attendees"],
    calendar_api_token_loc: str = "/app/gcp-api-token-data-team.pickle",
    session_id: str = None,
    use_session_id: bool = False,
    verbose: int = 1,
) -> dict:
    # Set up a calendar api connection and then download
    creds = get_credentials(token_loc=calendar_api_token_loc)
    service = build("calendar", "v3", credentials=creds)

    print("Downloading my calendar information within the date range")
    # now = datetime.datetime.utcnow().isoformat() + "Z"  # 'Z' indicates UTC time

    date_range = [datetime.strptime(x, "%Y-%m-%d").isoformat() + "Z" for x in date_range]

    # start_date = datetime.datetime(2023, 10, 10, 0, 0).isoformat() + "Z"
    # end_date = datetime.datetime(2023, 10, 31, 0, 0).isoformat() + "Z"
    events_result = (
        service.events()
        .list(
            calendarId="primary",
            timeMin=date_range[0],
            timeMax=date_range[1],
            maxResults=10_000,
            singleEvents=True,
            orderBy="startTime",
        )
        .execute()
    )
    events = events_result.get("items", [])

    # Summarize my calendar and select a few fields
    calendar_summaries = summarize_calendar(events)
    if calendar_fields is not None:
        calendar_summaries = [{k: d[k] for k in calendar_fields if k in d} for d in calendar_summaries]
    print("First event in calendar_summaries:\n%s" % json.dumps(calendar_summaries[0], indent=4))
    print("Last event in calendar_summaries:\n%s" % json.dumps(calendar_summaries[-1], indent=4))

    full_prompt = prompt_template.format(calendar_summaries=calendar_summaries)
    print(f"prompt_template is: {prompt_template}")

    if not use_session_id:
        session_id = None
    response = chat_with_backoff(
        model=gpt_model, messages=[{"role": "user", "content": full_prompt}], session_id=session_id
    )
    reply = response["choices"][0]["message"]["content"]
    print("Reply: %s" % reply)

    result = {"reply": reply, "session_id": response["id"]}
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpt_model")
    parser.add_argument("--prompt_template")
    parser.add_argument("--calendar_api_token_loc", default="/tmp/gcp-api-token-data-team.pickle")
    parser.add_argument("--session_id", default=None)
    args = vars(parser.parse_args())

    print("Command line args:\n%s" % json.dumps(args, indent=4))
    main(**args)
    print("ALL DONE!\n")
