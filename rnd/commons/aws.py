import base64
import boto3
from io import StringIO
import json
import pandas as pd
import os


def get_aws_client(
    service_name: str = "s3",
    region_name: str = "us-west-2",
    access_key: str = os.environ.get("AWS_ACCESS_KEY"),
    secret_key: str = os.environ.get("AWS_SECRET_KEY"),
    **kwargs,
):
    """Convenient function to create aws client. Use access key if exists, otherwise use the default setting"""
    client = boto3.client(
        service_name, region_name=region_name, aws_access_key_id=access_key, aws_secret_access_key=secret_key, **kwargs
    )
    return client


def get_aws_resource(
    service_name: str = "s3",
    region_name: str = "us-west-2",
    access_key: str = os.environ.get("AWS_ACCESS_KEY"),
    secret_key: str = os.environ.get("AWS_SECRET_KEY"),
    **kwargs,
):
    """Convenient function to create an aws resource. Use access key if exists, otherwise use the default setting"""
    res = boto3.resource(
        service_name, region_name=region_name, aws_access_key_id=access_key, aws_secret_access_key=secret_key, **kwargs
    )
    return res


def get_aws_session(
    region_name: str = "us-west-2",
    access_key: str = os.environ.get("AWS_ACCESS_KEY"),
    secret_key: str = os.environ.get("AWS_SECRET_KEY"),
    **kwargs,
):
    """Convenient function to create an aws session. Use access key if exists, otherwise use the default setting"""
    session = boto3.session.Session(
        region_name=region_name, aws_access_key_id=access_key, aws_secret_access_key=secret_key, **kwargs
    )
    return session


def get_aws_secret(secret_id: str, aws_session) -> dict:
    """Get an aws secret from Secrets Manager

    Args:
        secret_id (str): id to the secret
        aws_client: An AWS client that has access to the secrets manager
    Examples:
    """
    response = {}
    try:
        client = aws_session.client(service_name="secretsmanager")
        response = client.get_secret_value(SecretId=secret_id)
    except Exception as e:
        logger.error(f"get_aws_secret failed w/ error mesg: {str(e)}")

    # Decrypt the secret
    if "SecretString" in response:
        secret = response.get("SecretString")
    elif "SecretBinary" in response:
        secret = base64.b64decode(response.get("SecretBinary"))
    else:
        raise KeyError("get_aws_secret did not return a meaningful response: %s" % str(response))

    return json.loads(secret)


class S3FileUtils(object):
    """A list of utility functions dealing with S3 files"""

    def __init__(
        self, access_key: str = os.environ.get("AWS_ACCESS_KEY"), secret_key: str = os.environ.get("AWS_SECRET_KEY")
    ):
        self.access_key = access_key
        self.secret_key = secret_key

    def list_s3_files(self, bucket: str, prefix: str = "", delimiter: str = "/", **kwargs):
        """Listing all the files inside an s3 directory even if the file count is over 1000"""
        client = get_aws_client("s3", access_key=self.access_key, secret_key=self.secret_key)
        kwargs = dict(MaxKeys=1000, **kwargs)
        continuation_token = None
        while True:
            if continuation_token:  # Update the token
                kwargs["ContinuationToken"] = continuation_token

            response = client.list_objects_v2(Bucket=bucket, Prefix=prefix, Delimiter=delimiter, **kwargs)
            for obj in response.get("Contents"):
                fullpath = f"s3://{bucket}/{obj['Key']}"
                yield fullpath

            if not response.get("IsTruncated"):  # If at the end of the list
                break
            continuation_token = response.get("NextContinuationToken")
        return

    def list_s3_folders(self, bucket: str, prefix: str = "", delimiter: str = "/") -> str:
        """List the folders under a bucket and prefix

        Args:
            bucket: bucket name, e.g. "a" of "s3://a/b/"
            prefix: the directory path following the bucket name. e.g. "b/" of "s3://a/b/"
        """
        assert prefix == "" or prefix[-1] == "/", f"prefix must end with '/' but '{prefix}' doesn't."

        client = get_aws_client("s3", access_key=self.access_key, secret_key=self.secret_key)
        response = client.list_objects_v2(Bucket=bucket, Prefix=prefix, Delimiter=delimiter)
        for obj in response.get("CommonPrefixes"):
            # Do not keep the last '/'
            assert obj["Prefix"][-1] == "/", "It must end with '/' but it doesn't."
            fullpath = f"s3://{bucket}/{obj['Prefix'][:-1]}"
            yield fullpath

    def separate_bucket_and_path(self, full_path, verbose: int = 1):
        assert full_path.startswith("s3://"), f"{full_path} is not a valid s3 full_path"
        full_path = full_path.replace("s3://", "")
        loc = full_path.find("/")
        assert loc != -1, f"{full_path} contains no '/' which is incorrect"

        bkt = full_path[0:loc]
        path = full_path[loc + 1 :]
        if verbose >= 2:
            print(f"Separated bucket and path: '{bkt}' and '{path}'")
        return bkt, path

    def upload_file(self, filepath_in, filepath_s3):
        bkt, filepath = self.separate_bucket_and_path(filepath_s3)

        resource = get_aws_resource(service_name="s3")
        result = resource.Bucket(bkt).upload_file(filepath_in, filepath)
        print(f"Successfully copied {filepath_in} to {filepath_s3}: {result}")

    def copy_file_within_bucket(
        self, bucket: str, source_filepath: str, destination_filepath: str, copy_size_threshold: int = 5
    ):
        """
        Copy a file within the same AWS S3 bucket.

        Args:
            bucket (str): The name of the AWS S3 bucket that contains the file to be copied.
            source_filepath (str): The key of the file to be copied.
            destination_filepath (str): The key of the destination location for the file.

        Returns:
            None

        Raises:
            botocore.exceptions.NoCredentialsError: If valid AWS credentials are not found.
            botocore.exceptions.EndpointConnectionError: If there is an error connecting to the AWS S3 endpoint.
        """
        # Create an S3 client
        s3 = get_aws_client("s3", access_key=self.access_key, secret_key=self.secret_key)
        file_size = s3.get_object(Bucket=bucket, Key=source_filepath)["ContentLength"]  # size in bytes
        file_size_gb = file_size / 1024 / 1024 / 1024

        copy_source = {"Bucket": bucket, "Key": source_filepath}

        # Copy the object to the destination
        print(f"Copying the file: {source_filepath} into {destination_filepath}")

        if file_size_gb > copy_size_threshold:
            # this works for files greater than 5gb
            print(f"File is greater than {copy_size_threshold}GB")
            s3.copy(Bucket=bucket, CopySource=copy_source, Key=destination_filepath)
        else:
            s3.copy_object(Bucket=bucket, CopySource=copy_source, Key=destination_filepath)

    def move_file_within_bucket(self, bucket: str, source_filepath: str, destination_filepath: str):
        """
        Move a file within the same AWS S3 bucket.

        Args:
            bucket (str): The name of the AWS S3 bucket that contains the file to be moved.
            source_filepath (str): The key of the file to be moved.
            destination_filepath (str): The key of the destination location for the file.

        Returns:
            None

        Raises:
            botocore.exceptions.NoCredentialsError: If valid AWS credentials are not found.
            botocore.exceptions.EndpointConnectionError: If there is an error connecting to the AWS S3 endpoint.
        """

        print(f"Moving the file from {source_filepath} into {destination_filepath}")
        self.copy_file_within_bucket(
            bucket=bucket, source_filepath=source_filepath, destination_filepath=destination_filepath
        )
        self.delete_file(bucket, filepath=source_filepath)

    def delete_file(self, bucket: str, filepath: str):
        """
        Delete a file from a specified bucket.

        Args:
            bucket (str): The name of the AWS S3 bucket.
            filepath (str): The key of the file to be deleted.

        Returns:
            None

        Raises:
            botocore.exceptions.NoCredentialsError: If valid AWS credentials are not found.
            botocore.exceptions.EndpointConnectionError: If there is an error connecting to the AWS S3 endpoint.
        """
        s3 = get_aws_client("s3", access_key=self.access_key, secret_key=self.secret_key)
        print(f"Deleting the file: '{bucket}/{filepath}'")
        s3.delete_object(Bucket=bucket, Key=filepath)


class S3IO(object):
    def __init__(self):
        pass

    def to_json_s3(self, data: dict, dir: str, filename: str = "tmp.json", verbose: int = 1):
        """Uploading/saving a dict inside S3 directory
        Args:
            data:
            dir: in the format of "s3://<bucket>/<folder>"
            filename:
        """
        data_json = json.dumps(data, indent=4, default=str)
        bucket, folder_path = S3FileUtils().separate_bucket_and_path(dir)
        filepath = f"{folder_path}/{filename}"

        # Put the data inside s3 folder
        resource = get_aws_resource(service_name="s3")
        bucket_obj = resource.Bucket(name=bucket)
        bucket_obj.put_object(Key=filepath, Body=data_json)
        if verbose >= 2:
            print(f"Successfully saved json file at '{bucket}/{filepath}'")
        return

    def read_json_s3(self, dir: str, filename: str):
        """Reading a json inside S3 directory
        Args:
            dir: in the format of "s3://<bucket>/<folder>"
            filename:
        """
        bucket, folder_path = S3FileUtils().separate_bucket_and_path(dir)
        resource = get_aws_resource("s3")
        obj = resource.Object(bucket, key=f"{folder_path}/{filename}")
        data = obj.get()["Body"].read().decode("utf-8")
        return json.loads(data)

    def read_csv_s3(self, dir: str, filename: str):
        """Reading a csv file inside S3 directory
        Args:
            dir: in the format of "s3://<bucket>/<folder>"
            filename:
        """
        bucket, folder_path = S3FileUtils().separate_bucket_and_path(dir)
        resource = get_aws_resource("s3")
        obj = resource.Object(bucket, key=f"{folder_path}/{filename}")
        data = obj.get()["Body"].read().decode("utf-8")
        data = pd.read_csv(StringIO(data))
        return data
