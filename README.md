# bridgeAI-drift-monitoring

## Drift detection

1. The data used is available [here](https://www.kaggle.com/datasets/yasserh/housing-prices-dataset).
2. Ensure you have a model endpoint available that serves the regression model that we want to test the data against
3. Ensure the DRIFT_REPORT_BUCKET (named `evidently-reports`) is created and accessible
4. Update the python environment in `.env` file
5. Install `poetry` if not already installed
6. Install the dependencies using poetry `poetry install`
7. update the config and other parameters in the `config.yaml` file
8. Add `./src` to the `PYTHONPATH` - `export PYTHONPATH="${PYTHONPATH}:./src"`
9. Run `poetry run python src/main.py`

**The above manual steps are automated using the drift detection dag in the [DAGs repo](https://github.com/digicatapult/bridgeAI-airflow-DAGs) **\


### Environment Variables

The following environment variables need to be set for this repo.

| Variable                | Default Value                                                                  | Description                                                                                                   |
|-------------------------|--------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------|
| CONFIG_PATH             | `./config.yaml`                                                                | File path to the model training and other configuration file                                                  |
| LOG_LEVEL               | `INFO`                                                                         | The logging level for the application. Valid values are `DEBUG`, `INFO`, `WARNING`, `ERROR`, and `CRITICAL`.  |
| MLFLOW_TRACKING_URI     | `http://localhost:5000`                                                        | MLFlow tracking URI. Use `http://host.docker.internal:5000` if the MLFlow is running within docker container. |
| GITHUB_USERNAME         | None                                                                           | Githuib username. This is needed to pull the data form the dvc repo.                                          |
| GITHUB_PASSWORD         | None                                                                           | Githuib token. This is needed to pull the data form the dvc repo.                                             |
| DVC_ACCESS_KEY_ID       | `admin`                                                                        | Access key for dvc remote                                                                                     |
| DVC_SECRET_ACCESS_KEY   | `password`                                                                     | secret access key for dvc remote                                                                              |
| DVC_REMOTE_NAME         | `regression-model-remote`                                                      | A name assigned to the dvc remote                                                                             |
| DVC_REMOTE              | `s3://bridgeai-dvc-remote`                                                     | DVC remote path (to s3/minio bucket)                                                                          |
| DVC_ENDPOINT_URL        | `http://minio`                                                                 | Endpoint url for dvc remote                                                                                   |
| DATA_REPO               | `https://github.com/digicatapult/bridgeAI-regression-model-data-ingestion.git` | data ingestion repo where the data is versioned with dvc                                                      |
| HISTORICAL_DATA_VERSION | `data-v1.0.0`                                                                  | the data version (dvc tagged version from the data ingestion repo) used for training the model                |
| NEW_DATA_VERSION        | `data-v1.1.0`                                                                  | the data version (dvc tagged version from the data ingestion repo) curresponding to the new data              |
| MODEL_ENDPOINT          | `http://host.docker.internal:5001/invocations`                                 | deployed model endpoint using which predictions can be made                                                   |
| DRIFT_REPORT_BUCKET     | `bridgeai-evidently-reports`                                                   | s3 bucket name where the generated html report will be saved                                                  |


Note:
If you are using local kind cluster to create the inference service, you can follow below steps to find the correct `MODEL_ENDPOINT`.
1. identify the pod corresponding to the model inference service by running `kubectl get pods -n default`. If using a different namespace, replace the `default` with the right name space
2. Once you identify the pod, port forward the inference service `kubectl port-forward <pod name> 8081:8080 -n default`. Replace the pod name with the actual value which would look something like this `house-price-predictor-76f7659dfc-q6k78` 
3. Now the `MODEL_ENDPOINT` will be something like `http://localhost:8081/v2/models/house_price_prediction_prod/infer` where you can replace the `house_price_prediction_prod` with the registered model name that you have deployed using kserve
4. If you are using docker or DAG to run the scripts in this repo, you may need to replace the `localhost` with `host.docker.internal` as well

### Running the tests

Ensure that you have the project requirements already set up by following the [Data Ingestion and versioning](#data-ingestion-and-versioning) instructions
- Ensure `pytest` is installed. `poetry install` will install it as a dependency.
- Run the tests with `poetry run pytest ./tests`