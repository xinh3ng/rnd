"""Save calendar summaries

# Usage Example

write_mode=replace

verbose=3

python rnd/ai/calendar/serving/save_calendar_summaries.py --write_mode=$write_mode --verbose=$verbose

"""
from datetime import datetime
from googleapiclient.discovery import build
import json
import openai
import os
import pandas as pd
import sqlite3
from typing import List

from rnd.ai.calendar.utils.calendar_utils import get_credentials, summarize_calendar


pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 120)
pd.set_option("display.max_colwidth", None)  # No truncation


openai.api_key = os.environ.get("OPENAI_API_KEY")


class DbOperator(object):
    def __init__(self, db: str = "calendar.db"):
        self.db = db
        self.conn = sqlite3.connect(self.db)  # create or connect to a sqlite db

    def create_table(self, table: str = "calendar_summary"):
        # Create a new table
        self.conn.execute(
            f"""
        create table if not exists {table} (
            title TEXT,
            description TEXT       
        )
        """
        )

    def write(self, data: pd.DataFrame, table: str = "calendar_summary", write_mode: str = "replace"):
        write_modes_allowed = ["replace", "append"]
        assert write_mode in write_modes_allowed, f"write_mode must be among {write_modes_allowed}"
        _ = data.to_sql(name=table, con=self.conn, if_exists=write_mode, index=False)
        print(f"Successfully write data into '{table}' table w/ write_mode: '{write_mode}'")

    def read(self, sql_query: str):
        data = pd.read_sql(sql_query, self.conn)
        return data

    def close_connection(self):
        self.conn.close()


########################################


def proc_data(data: List[dict]) -> pd.DataFrame:
    data = pd.DataFrame.from_records(data)
    data["attendees"] = data["attendees"].apply(lambda x: json.dumps(x))
    return data


def main(
    date_range: List[str] = ["2023-01-01", "2023-12-31"],
    calendar_fields: List[str] = None,
    calendar_api_token_loc: str = "/app/gcp-api-token-data-team.pickle",
    write_mode: str = "append",
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
            maxResults=100_000,
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

    # Write into a data table
    calendar_summaries = proc_data(calendar_summaries)
    op = DbOperator()
    op.write(data=calendar_summaries, write_mode=write_mode)

    # Testing purpose
    result = op.read(sql_query="select * from calendar_summary limit 5")
    print("Test is a succees")
    if verbose >= 3:
        print("Examples:\n%s" % result.head(5).to_string(line_width=120))
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--calendar_api_token_loc", default="/tmp/gcp-api-token-data-team.pickle")
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--write_mode", default="append")
    args = vars(parser.parse_args())

    print("Command line args:\n%s" % json.dumps(args, indent=4))
    main(**args)
    print("ALL DONE!\n")
