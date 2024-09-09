import datetime
import os
from pathlib import Path

import boto3
from botocore.client import Config
from botocore.exceptions import ClientError

from src.utils import load_yaml_config

# Configuration details
config = load_yaml_config()

s3_endpoint = os.getenv(
    "DVC_ENDPOINT_URL"
)  # eg: "s3://minio/" or http://localhost:9000
access_key = os.getenv("DVC_ACCESS_KEY_ID")
secret_key = os.getenv("DVC_SECRET_ACCESS_KEY")
region = config["dvc"]["dvc_region"]

# Create a session and S3 client
s3_client = boto3.client(
    "s3",
    endpoint_url=s3_endpoint,
    aws_access_key_id=access_key,
    aws_secret_access_key=secret_key,
    config=Config(signature_version="s3v4"),
    region_name=region,
)


def create_bucket_if_not_exists(bucket_name, s3_client):
    """Create s3 bucket if it doesn't exist."""
    try:
        # List all existing buckets to check if the bucket exists
        buckets = s3_client.list_buckets()
        bucket_names = [bucket["Name"] for bucket in buckets["Buckets"]]
        if bucket_name in bucket_names:
            print(f"Bucket '{bucket_name}' already exists.")
        else:
            # Create the bucket without LocationConstraint for MinIO
            s3_client.create_bucket(Bucket=bucket_name)
            print(f"Bucket '{bucket_name}' created.")
    except ClientError as e:
        # Handle other errors
        print(f"Error occurred: {e}")


def upload(file_name):
    bucket_name = "evidently-reports"
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # Adding the timestamp to the filename
    object_name = f"report_{timestamp}.html"

    create_bucket_if_not_exists(bucket_name, s3_client)

    # Upload the file
    try:

        s3_client.upload_file(str(file_name), bucket_name, object_name)
        print(
            f"File {file_name} uploaded to "
            f"bucket {bucket_name} as {object_name}."
        )
    except Exception as e:
        print(f"Error uploading file: {e}")


if __name__ == "__main__":
    # load the config file
    config = load_yaml_config()
    report_save_path = Path(config["report_save_path"]).resolve()
    upload(report_save_path)
