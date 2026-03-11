import mlflow
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv

load_dotenv(".env")
client = MlflowClient(os.getenv("MLFLOW_TRACKING_URI"))
run_id = "25368db90aaf41e1aa92e94cb8845810"
run = client.get_run(run_id)
print(f"Physical Path: {run.info.artifact_uri}")

# This will attempt to pull the files to your local 'test_download' folder
local_path = client.download_artifacts(run_id, "model_pipeline", dst_path="./test_download")
print(f"Files found at: {local_path}")
