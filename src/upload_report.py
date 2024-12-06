import datetime
import os
from pathlib import Path

import boto3
from botocore.client import Config

from src.utils import load_yaml_config


def get_s3_client():
    """Create and return an S3 client."""
    s3_endpoint = os.getenv("DVC_ENDPOINT_URL")
    access_key = os.getenv("DVC_ACCESS_KEY_ID")
    secret_key = os.getenv("DVC_SECRET_ACCESS_KEY")
    region = os.getenv("AWS_DEFAULT_REGION")

    if access_key is None or access_key == "":
        boto3.client(
            "s3",
            endpoint_url=s3_endpoint,
            config=Config(signature_version="s3v4"),
            region_name=region,
        )
    else:
        return boto3.client(
            "s3",
            endpoint_url=s3_endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            config=Config(signature_version="s3v4"),
            region_name=region,
        )


def upload(s3_client, file_name, bucket_name):
    """Upload the report file to the s3 bucket."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # Adding the timestamp to the filename when saving it
    object_name = f"report_{timestamp}.html"

    # Upload the file
    try:
        s3_client.upload_file(str(file_name), bucket_name, object_name)
        print(
            f"File {file_name} uploaded to "
            f"bucket {bucket_name} as {object_name}."
        )
    except Exception as e:
        print(f"Error uploading file: {e}")
        raise e


if __name__ == "__main__":
    # load the config file
    config = load_yaml_config()
    report_save_path = Path(config["report_save_path"]).resolve()
    bucket_name = os.getenv(
        "DRIFT_REPORT_BUCKET", config["report_save_bucket"]
    )
    s3_client = get_s3_client()
    upload(s3_client, report_save_path, bucket_name)
