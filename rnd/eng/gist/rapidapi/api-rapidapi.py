"""
https://towardsdatascience.com/develop-and-sell-a-python-api-from-start-to-end-tutorial-9a038e433966
"""
import os
import pandas as pd
import requests


def download(url: str, dest_folder: str):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    filename = url.split("/")[-1].replace(" ", "_")
    file_path = os.path.join(dest_folder, filename)
    r = requests.get(url, stream=True)
    if r.ok:
        print("saving to", os.path.abspath(file_path))
        with open(file_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 8):
                if chunk:
                    f.write(chunk)
                    f.flush()
                    os.fsync(f.fileno())
    else:
        print("Download failed: status code {}\n{}".format(r.status_code, r.text))


# url_to_titanic_data = "https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv"
# download(url_to_titanic_data, "./data")


df = pd.read_csv("./data/titanic.csv")
df.to_json(r"./data/titanic.json")
