"""Checking user retentions and churns in SQL

You're given the following table showing user logins to your system: 

Using the table above: Write a SQL query that returns the number of retained users per month in 2021. In this case, we'll define retentions for each given month as the number of users who logged in that month and also logged in the prior month. 

Once you've written that query, write one additional query to return the users who did not come back in a given month (e.g. they logged in the month prior, but not the next consecutive month) -- we'll call these our churned users.

"""
import datetime
import dateutil
import json
import pandas as pd
import random

small_data = pd.DataFrame(
    {
        "user_id": [1, 2, 3, 4, 1, 2, 3],
        "month": [1, 1, 1, 1, 2, 3, 2],
        "year": [2021, 2021, 2021, 2021, 2021, 2021, 2021],
    }
)


def create_big_data(rows: int, users: int):
    """Generates a random dataframe of data

    Args:
        rows: total number of
    """
    month = []
    user_id = []
    year = []

    for _ in range(rows):
        month.append(random.randint(1, 12))
        user_id.append(random.randint(1, users))
        year.append(2021)

    data = pd.DataFrame({"user_id": user_id, "month": month, "year": year}).sort_values(
        by=["user_id", "year", "month"], ignore_index=True
    )
    print("big_data examples:\n%s\n" % (data))
    return data


def calc_retentions(data: pd.DataFrame, verbose: int = 1):
    """"""
    data = data.copy(deep=True)
    # Create necessary data set
    newdata = data.groupby(["user_id"]).shift(-1)  # This is the key step
    data.rename(columns={"year_mon": "prev_year_mon"}, inplace=True)
    data["year_mon"] = newdata["year_mon"]

    retentions = pd.DataFrame()
    for m in pd.date_range(data["year_mon"].min(), data["year_mon"].max(), freq="MS"):
        # Retention means both this and previous months
        filtered = data[
            (data["year_mon"] == m) & (data["prev_year_mon"] == m - dateutil.relativedelta.relativedelta(months=1))
        ]
        row = pd.DataFrame.from_dict({"year_mon": [m], "retained": [len(filtered)]})
        retentions = pd.concat([retentions, row], ignore_index=True)
    if verbose >= 2:
        print("calc_retentions' result:\n%s\n" % (retentions.to_string(line_width=120)))
    return retentions


def calc_churns(data: pd.DataFrame, verbose: int = 1):
    """"""
    data = data.copy(deep=True)
    # Create necessary data set
    newdata = data.groupby(["user_id"]).shift(-1)
    data.rename(columns={"year_mon": "prev_year_mon"}, inplace=True)
    data["year_mon"] = newdata["year_mon"]

    churns = pd.DataFrame()
    for m in pd.date_range(data["year_mon"].min(), data["year_mon"].max(), freq="MS"):
        # Retention means both this and previous months
        filtered = data[
            (data["year_mon"] != m) & (data["prev_year_mon"] == m - dateutil.relativedelta.relativedelta(months=1))
        ]
        row = pd.DataFrame.from_dict({"year_mon": [m], "retained": [len(filtered)]})
        churns = pd.concat([churns, row], ignore_index=True)
    if verbose >= 2:
        print("calc_churns's result:\n%s\n" % (churns.to_string(line_width=120)))
    return churns


def add_year_mon(data: pd.DataFrame):
    """Add year-mon so that I can deal with incrase/decrease by a month, etc."""
    data["year_mon"] = pd.to_datetime(
        data["year"].astype(str) + "-" + data["month"].astype(str), format="%Y-%B", infer_datetime_format=True
    )
    data.drop(["year", "month"], axis=1, inplace=True)
    return data


def main(small_or_big: bool, rows: int = 100, users: int = 10):
    if small_or_big == "big":
        data = create_big_data(rows=rows, users=users)
    elif small_or_big == "small":
        data = small_data

    print("data:\n%s\n" % data)
    data = add_year_mon(data)

    calc_retentions(data=data, verbose=2)
    calc_churns(data=data, verbose=2)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--small_or_big", default="small")

    args = vars(parser.parse_args())
    print("Cmd line args:\n{}".format(json.dumps(args, sort_keys=True, indent=4)))

    main(**args)
    print("ALL DONE!\n")
