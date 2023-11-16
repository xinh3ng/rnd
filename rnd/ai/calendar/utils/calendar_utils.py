from google_auth_oauthlib.flow import InstalledAppFlow
import os
import pickle
from typing import List

from xhcaftv.commons.commons import create_logger
from xhcaftv.commons import aws


logger = create_logger(__name__)


def get_credentials(
    token_loc: str = "/tmp/gcp-api-token-data-team.pickle",
    scopes: List[str] = ["https://www.googleapis.com/auth/calendar.readonly"],
):
    """Get credentials

    To avoid repeated authentication, we save the credentials in /tmp folder
    """
    if os.path.exists(token_loc):
        with open(token_loc, "rb") as f:
            credentials = pickle.load(f)
        return credentials

    aws_session = aws.get_aws_session(region_name="us-west-2")
    secret = aws.get_aws_secret("gcp-creds-data-team", aws_session)
    flow = InstalledAppFlow.from_client_config(client_config=secret, scopes=scopes)
    credentials = flow.run_local_server(port=0)
    with open(token_loc, "wb") as f:  # Save the credentials
        pickle.dump(credentials, f)
    return credentials


def summarize_calendar(events: List):
    # Simplified parsing of the events data to a structure that can be easily understood by the model.
    calendar_summaries = []
    for event in events:
        summary = {
            "title": event.get("summary", "busy"),
            "description": event.get("description", ""),
            "start": event["start"].get("dateTime", event["start"].get("date")),
            "end": event["end"].get("dateTime", event["end"].get("date")),
            "organizer": event.get("organizer", {}).get("email", ""),
            "attendees": [attendee["email"] for attendee in event.get("attendees", [])],
        }
        calendar_summaries.append(summary)
    return calendar_summaries
