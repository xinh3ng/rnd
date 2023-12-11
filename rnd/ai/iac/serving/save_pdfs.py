"""Save PDFs that I downloaded from IAC website

The processed data is saved as
year | subject | grade | question | answer


# Usage Example

write_mode=overwrite

verbose=3

python rnd/ai/iac/serving/save_pdfs.py

"""
from cafpyutils.generic import create_logger
import fitz  # PyMuPDF
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import io
import json
import os
import pandas as pd
import sqlite3
from typing import List


logger = create_logger(__name__)

gcp_scopes = ["https://www.googleapis.com/auth/drive.metadata.readonly", "https://www.googleapis.com/auth/drive"]

# openai.api_key = os.environ.get("OPENAI_API_KEY")

pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 120)
pd.set_option("display.max_colwidth", None)  # No truncation


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


def get_credentials(
    token_loc: str = f"{os.environ['HOME']}/.secrets/googlecloud/xin.heng@gmail.com/token.json",
    credentials_json: str = f"{os.environ['HOME']}/.secrets/googlecloud/xin.heng@gmail.com/credentials.json",
    scopes: List[str] = gcp_scopes,
):
    """Get credentials

    To avoid repeated authentication, we save the credentials in /tmp folder
    """
    creds = None
    if os.path.exists(token_loc):
        creds = Credentials.from_authorized_user_file(token_loc, scopes)

    # If there are no (valid) credentials available, let the user log in
    if (creds is None) or (not creds.valid):
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(credentials_json, scopes)
            creds = flow.run_local_server(port=0)

    with open(token_loc, "w") as token:
        token.write(creds.to_json())

    return creds


def download_file(service, file_id):
    """Download file with Google Drive ID"""

    request = service.files().get_media(fileId=file_id)
    file = io.BytesIO()
    downloader = MediaIoBaseDownload(file, request)
    done = False
    while not done:
        status, done = downloader.next_chunk()
        print("Download progress: {0}".format(status.progress() * 100))
    file.seek(0)
    return file


def extract_text_from_pdf(file_stream):
    document = fitz.open("pdf", file_stream)
    text = ""
    for page in document:
        text += page.get_text()
    document.close()
    return text


########################################


def main(
    date_range: str = None,
    write_mode: str = "overwrite",
    verbose: int = 1,
) -> dict:
    """Main function"""
    date_range = ["2000", "2099"] if date_range is None else date_range.split(",")

    creds = get_credentials(scopes=gcp_scopes)
    service = build("drive", "v3", credentials=creds)

    # https://drive.google.com/drive/folders/14ZLljcELtt_eQ2gC6UF_VGUM7pi1sA_q
    folder_id = "14ZLljcELtt_eQ2gC6UF_VGUM7pi1sA_q"

    files, page_token = [], None
    query = f"'{folder_id}' in parents"

    # Query to get files from the specific folder
    response = service.files().list(q=query, pageSize=20, fields="nextPageToken, files(id, name)").execute()

    items = response.get("files", [])
    # for item in items:

    for item in items:
        print(f"""Reading filename: '{item["name"]}' with ID: {item['id']}""")
        file_stream = download_file(service, file_id=item["id"])
        text = extract_text_from_pdf(file_stream)
        logger.info(text[:500])

    breakpoint()

    op.write(data=data, write_mode=write_mode)

    # Testing purpose
    result = op.read(sql_query="select * from calendar_summary limit 5")
    print("Test is a succees")
    if verbose >= 3:
        print("Examples:\n%s" % result.head(5).to_string(line_width=120))
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--write_mode")
    parser.add_argument("--verbose", type=int, default=1)

    args = vars(parser.parse_args())
    logger.info("Command line args:\n%s" % json.dumps(args, indent=4))
    main(**args)
    logger.info("ALL DONE!\n")
