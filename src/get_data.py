"""Get the data to be compared for drift detection from the feature store."""

import os
import shutil
from pathlib import Path

from dvc.cli import main as dvc_main
from git import Repo

from src.utils import load_yaml_config, logger


def checkout_data(repo, data_version):
    """Checkout to the data branch."""
    repo.git.fetch()
    try:
        repo.git.checkout(data_version)
    except Exception as e:
        logger.error(
            f"Git data version-{data_version} checkout failed with error: {e}"
        )
        raise e


def get_authenticated_github_url(base_url):
    """From the base git http url, generate an authenticated url."""
    username = os.getenv("GITHUB_USERNAME")
    password = os.getenv("GITHUB_PASSWORD")

    if not username or not password:
        logger.error(
            "GITHUB_USERNAME or GITHUB_PASSWORD environment variables not set"
        )
        raise ValueError(
            "GITHUB_USERNAME or GITHUB_PASSWORD environment variables not set"
        )

    # Separate protocol and the rest of the URL
    protocol, rest_of_url = base_url.split("://")

    # Construct the new URL with credentials
    new_url = f"{protocol}://{username}:{password}@{rest_of_url}"

    return new_url


def delete_file_if_exists(file_path):
    """Delete a file if it exists."""
    if os.path.exists(file_path):
        os.remove(file_path)


def delete_directory_if_exists(directory_path):
    """Delete a directory and all its contents if it exists."""
    if os.path.exists(directory_path) and os.path.isdir(directory_path):
        shutil.rmtree(directory_path)


def dvc_pull(config):
    """DVC pull."""
    # first remove if older data exists
    csv_files = [
        f for f in os.listdir(config["dvc"]["data_path"]) if f.endswith(".csv")
    ]
    dvc_remote_name = os.getenv(
        "DVC_REMOTE_NAME", config["dvc"]["dvc_remote_name"]
    )
    for file in csv_files:
        delete_file_if_exists(file)
    try:
        dvc_remote_add(config)
        dvc_main(["pull", "-r", dvc_remote_name])
    except Exception as e:
        logger.error(f"DVC pull failed with error: {e}")
        raise e


def dvc_remote_add(config):
    """Set the dvc remote."""
    access_key_id = os.getenv("DVC_ACCESS_KEY_ID")
    secret_access_key = os.getenv("DVC_SECRET_ACCESS_KEY")
    region = os.getenv("AWS_DEFAULT_REGION")
    try:
        dvc_remote_name = os.getenv(
            "DVC_REMOTE_NAME", config["dvc"]["dvc_remote_name"]
        )
        dvc_remote = os.getenv("DVC_REMOTE", config["dvc"]["dvc_remote"])
        dvc_endpoint_url = os.getenv(
            "DVC_ENDPOINT_URL", config["dvc"]["dvc_endpoint_url"]
        )
        dvc_main(["remote", "add", "-f", dvc_remote_name, dvc_remote])
        dvc_main(
            [
                "remote",
                "modify",
                dvc_remote_name,
                "endpointurl",
                dvc_endpoint_url,
            ]
        )
        if secret_access_key is None or secret_access_key == "":
            # Set dvc remote credentials
            # only when a valid secret access key is present
            logger.warning(
                "AWS credentials `dvc_secret_access_key` is missing "
                "in the Airflow connection.\n"
                "Falling back to IAM based s3 authentication."
            )
        else:
            dvc_main(
                [
                    "remote",
                    "modify",
                    dvc_remote_name,
                    "access_key_id",
                    access_key_id,
                ]
            )
            dvc_main(
                [
                    "remote",
                    "modify",
                    dvc_remote_name,
                    "secret_access_key",
                    secret_access_key,
                ]
            )
        # Minio does not enforce regions but DVC requires it
        dvc_main(["remote", "modify", dvc_remote_name, "region", region])
    except Exception as e:
        logger.error(f"DVC remote add failed with error: {e}")
        raise e


def move_dvc_data(source_repo, save_path):
    """Move pulled dvc data to where it is expected to be."""
    # First delete if the destination has files with same name
    delete_file_if_exists(save_path)

    # Only using the train split of the data here
    # TODO: modify if want to include all the splits of data

    # Now move the data to destination
    try:
        shutil.move("./artefacts/train_data.csv", save_path)
    except Exception as e:
        logger.error(f"Copying dvc data failed with error {e}")
        raise e


def fetch_data(config, data_version, save_path):
    """Fetch the versioned data from dvc."""
    # 1. Authenticate, clone, and update git repo
    data_repo = os.getenv("DATA_REPO", config["dvc"]["git_repo_url"])

    authenticated_git_url = get_authenticated_github_url(data_repo)
    repo_temp_path = "./repo"
    Repo.clone_from(authenticated_git_url, repo_temp_path)

    os.chdir(repo_temp_path)

    # 2. Initialise git and dvc
    repo = Repo("./")
    assert not repo.bare

    # 3. Checkout to the data version
    checkout_data(repo, data_version)

    # 4. DVC pull
    dvc_pull(config)

    # 5. move the pulled data to expected location
    move_dvc_data(config, save_path)
    os.chdir("../")
    delete_directory_if_exists(repo_temp_path)


if __name__ == "__main__":
    # load the config file
    config = load_yaml_config()

    historical_data_version = os.getenv(
        "HISTORICAL_DATA_VERSION", config["historical_data_version"]
    )
    new_data_version = os.getenv(
        "NEW_DATA_VERSION", config["new_data_version"]
    )

    historical_data_save_path = Path(
        config["historical_data_save_path"]
    ).resolve()
    new_data_save_path = Path(config["new_data_save_path"]).resolve()

    fetch_data(config, historical_data_version, historical_data_save_path)
    fetch_data(config, new_data_version, new_data_save_path)
