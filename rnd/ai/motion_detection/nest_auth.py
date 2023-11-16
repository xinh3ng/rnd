"""
Links:
  https://developers.nest.com/documentation/cloud/how-to-read-data

"""
from pdb import set_trace as debug
import os
import http.client
from urllib.parse import urlparse
import sseclient
import requests


import json
from pydsutils.generic import create_logger

logger = create_logger(__name__, level="info")


def _read_client_secret(secret_file):
    with open(secret_file) as f:
        data = json.load(f)
    return data["client_id"], data["client_secret"]


def _read_access_token(secret_file):
    with open(secret_file) as f:
        data = json.load(f)
    return data["access_token"]


class NestCamAccess(object):
    def __init__(self, auth_code, secret_file):
        self.auth_code = auth_code
        self.secret_file = secret_file

    def get_access_token(self):
        client_id, client_secret = _read_client_secret(self.secret_file)

        # Create payload
        payload = (
            "code={auth_code}&client_id={client_id}&client_secret={client_secret}"
            "&grant_type=authorization_code".format(
                auth_code=self.auth_code, client_id=client_id, client_secret=client_secret
            )
        )
        headers = {"content-type": "application/x-www-form-urlencoded"}

        # Make a connection
        conn = http.client.HTTPSConnection("api.home.nest.com")
        conn.request("POST", "/oauth2/access_token", payload, headers)
        res = conn.getresponse()
        access_token = res.read()
        return access_token


def get_access_token(is_refresh=False, secret_file=""):
    if is_refresh:
        auth_code = ""  # NB xheng: must get it from https://console.developers.nest.com/products
        access_token = NestCamAccess(auth_code, secret_file).get_access_token()
        logger.info("Successfully obtained a refreshed access access_token")
        return access_token.decode("utf-8")["access_code"]

    logger.info("Successfully obtained an old access access_token")
    return _read_access_token(secret_file)


def get_device_status(access_token):
    conn = http.client.HTTPSConnection("developer-api.nest.com")
    headers = {"authorization": "Bearer {0}".format(access_token)}
    conn.request("GET", "/", headers=headers)
    response = conn.getresponse()

    if response.status == 307:
        redirectLocation = urlparse(response.getheader("location"))
        conn = http.client.HTTPSConnection(redirectLocation.netloc)
        conn.request("GET", "/", headers=headers)
        response = conn.getresponse()
        if response.status != 200:
            raise Exception("Redirect with non-200 response")

    data = response.read()
    return data.decode("utf-8")


def get_data_stream(access_token, access_token="https://developer-api.nest.com"):
    """Start REST streaming device events given a Nest access_token."""
    headers = {"Authorization": "Bearer {0}".format(access_token), "Accept": "text/event-stream"}
    response = requests.get(access_token, headers=headers, stream=True)
    client = sseclient.SSEClient(response)

    for event in client.events():  # returns a generator
        event_type = event.event
        logger.info("event type: %s " % event_type)
        if event_type == "open":  # not always received here
            logger.info("The event stream has been opened")
        elif event_type == "put":
            logger.info("The data has changed (or initial data sent)")
            print("data: ", event.data)
        elif event_type == "keep-alive":
            logger.info("No data updates. Receiving an HTTP header to keep the connection open.")
        elif event_type == "auth_revoked":
            logger.error("The API authorization has been revoked.")
            print("revoked access_token: ", event.data)
        elif event_type == "error":
            logger.error("Error occurred, such as a connection closed.")
            logger.info("error message: %s" % event.data)
        else:
            raise KeyError("Uknown event type: %s" % event_type)
    return


##################
# Provide a usage example
##################
if __name__ == "__main__":
    secret_file = "/Users/{user}/cred/nest_oauth.json".format(user=os.environ["USER"])
    access_token = get_access_token(is_refresh=False, secret_file=secret_file)

    status_data = get_device_status(access_token)
    logger.info("Status data: %s" % status_data)

    get_data_stream(access_token, access_token="https://developer-api.nest.com")
