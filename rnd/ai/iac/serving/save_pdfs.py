"""Save PDFs that I downloaded from IAC website

The processed data is saved as
year | subject |      geo | grade | question_no | question | answer | filename
2014 | history | national |   "4" |           1 |      ... |  Japan | ...
...
...

####################
# Usage Example
####################

# folder_id: 
# "14ZLljcELtt_eQ2gC6UF_VGUM7pi1sA_q", history folder
# "1jB8RaTbwdr09gUukcq-orS4NsFHR-9f2", geography folder 
# "1urBwsQvb4_lEG0mdP-QH7AtrsY9ok4Mb", 
folder_id="1jB8RaTbwdr09gUukcq-orS4NsFHR-9f2"

write_mode=replace

verbose=3

python rnd/ai/iac/serving/save_pdfs.py --folder_id=$folder_id --write_mode=$write_mode --verbose=$verbose

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
import re
from typing import List


logger = create_logger(__name__)

gcp_scopes = ["https://www.googleapis.com/auth/drive.metadata.readonly", "https://www.googleapis.com/auth/drive"]

# openai.api_key = os.environ.get("OPENAI_API_KEY")

pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 120)
pd.set_option("display.max_colwidth", None)  # No truncation


class DbOperator(object):
    def __init__(self, db: str):
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

    def write(self, data: pd.DataFrame, table: str, write_mode: str = "replace", verbose: int = 1):
        write_modes_allowed = ["replace", "append"]
        assert write_mode in write_modes_allowed, f"write_mode must be among {write_modes_allowed}"
        _ = data.to_sql(name=table, con=self.conn, if_exists=write_mode, index=False)
        logger.info(f"Successfully write into '{table}' table w/ write_mode: '{write_mode}'")

    def read_as_pandas(self, sql_query: str) -> pd.DataFrame:
        data = pd.read_sql(sql_query, self.conn)
        return data

    def read(sel, sql_query: str) -> str:
        raise NotImplementedError

    def close_connection(self):
        self.conn.close()


########################################


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
        # logger.info("Download progress: {0}".format(status.progress() * 100))
    file.seek(0)
    return file


def extract_text_from_pdf(file_stream) -> str:
    document = fitz.open("pdf", file_stream)
    text = ""
    for page in document:
        text += page.get_text()
    document.close()
    return text


def extract_fields_from_text(text: str, verbose: int = 1) -> dict:
    """The input text contains the entire pdf file. We tease out all the useful information

    Sample string
        large_string = "
        (1) What is the capital of France?
        Answer: Paris
        (2) What is the largest mammal?
        Answer: Blue Whale
        ... (and so on) ...
        (100) What is the smallest planet in our solar system?
        Answer: Mercury
        "
    """

    # Regular expression pattern
    # Explanation:
    # \(\d+\) - Matches a number in parentheses (e.g., (1), (2), etc.)
    # \((\d+)\) - Matches and captures a number in parentheses (e.g., (1), (2), etc.)
    # [\s]* - Matches any whitespace character (like space, newline) zero or more times
    # .+? - Matches any characters (non-greedily) up to the next part of the pattern
    # ANSWER: - Matches the literal string "Answer:"
    # .+? - Matches the answer text (non-greedily)
    # (?=\(\d+\)|$) - Positive lookahead to ensure that each answer is followed by another question number or the end of the string

    # All possible patterns
    patterns = [
        r"\((\d+)\)[\s]*(.+?)\nANSWER: (.+?)\n(?=\(\d+\)|$)",
        r"(\d+)\. (.+?)\nANSWER: (.+?)\n(?=\d+\. |$)",
        r"(\d+)\.[\s]*(.+?)\nANSWER: (.+?)\n(?=\d+\.|$)",
    ]

    for pattern in patterns:
        pairs = re.findall(pattern, text, re.DOTALL)
        if len(pairs) == 0:  # Now to try a different pattern
            continue

        fields = []
        for idx, (question_no, question, answer) in enumerate(pairs, 1):
            j = {"question_no": question_no, "question": question, "answer": answer}
            fields.append(j)
        break

    logger.info(f"Successfully extracted {idx + 1} pairs of questions and answers")
    if idx + 1 < 20:
        logger.warning("Didn't seem to extract enough pairs")

    data = pd.DataFrame(fields)
    return data


########################################


def main(
    folder_id: str,
    write_mode: str = "replace",
    verbose: int = 1,
) -> dict:
    """Main function"""

    creds = get_credentials(scopes=gcp_scopes)
    service = build("drive", "v3", credentials=creds)

    # Querying files from the specific folder
    # https://drive.google.com/drive/folders/14ZLljcELtt_eQ2gC6UF_VGUM7pi1sA_q
    query = f"'{folder_id}' in parents"
    response = service.files().list(q=query, pageSize=20, fields="nextPageToken, files(id, name)").execute()
    results = []
    for item in response.get("files", []):
        logger.info(f"""Reading filename: '{item["name"]}' with ID: {item['id']}""")
        file_stream = download_file(service, file_id=item["id"])
        text = extract_text_from_pdf(file_stream)
        results.append(extract_fields_from_text(text))

    data = pd.concat(results)
    logger.info("data examples:\n%s", data.head(3).to_string(line_width=120))
    [int(x) for x in set(data.question_no)]  # All question numbers must be able to convert into integer

    # Save in sqlite3
    op = DbOperator(db="bees.db")
    op.write(data=data, table="qa", write_mode=write_mode, verbose=verbose)
    result = op.read_as_pandas(sql_query="select * from qa limit 5")  # Validation purpose
    assert len(result) >= 1

    # Save in csv
    data.to_csv("/tmp/bees_qa.csv", index=False)
    return


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--folder_id")
    parser.add_argument("--write_mode", default="replace")
    parser.add_argument("--verbose", type=int, default=1)

    args = vars(parser.parse_args())
    logger.info("Command line args:\n%s" % json.dumps(args, indent=4))
    main(**args)
    logger.info("ALL DONE!\n")
