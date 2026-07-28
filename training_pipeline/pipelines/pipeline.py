from pathlib import Path
import subprocess
import sys
import os


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent

RAW_INPUT_PATH = PROJECT_ROOT / "data" / "raw" / "telco_churn.csv"
INGESTED_OUTPUT_PATH = PROJECT_ROOT / "training_pipeline" / "data" / "raw" / "data.csv"


def run_stage(stage_name, command):
	print(f"\n[{stage_name}] Starting")
	print("Command:", " ".join(str(part) for part in command))
	env = os.environ.copy()
	env["MLFLOW_ALLOW_FILE_STORE"] = "true"
	subprocess.run(command, check=True, cwd=PROJECT_ROOT, env=env)
	print(f"[{stage_name}] Completed")


def main():
	stages = [
		(
			"data_ingestion",
			[
				sys.executable,
				str(PROJECT_ROOT / "training_pipeline" / "components" / "data_ingestion.py"),
				"--input_path",
				str(RAW_INPUT_PATH),
				"--output_path",
				str(INGESTED_OUTPUT_PATH),
			],
		),
		(
			"data_preprocessing",
			[
				sys.executable,
				str(PROJECT_ROOT / "training_pipeline" / "components" / "data_preprocessing.py"),
			],
		),
		(
			"model_training",
			[
				sys.executable,
				str(PROJECT_ROOT / "training_pipeline" / "components" / "model_training.py"),
			],
		),
	]

	for stage_name, command in stages:
		run_stage(stage_name, command)

	print("\nPipeline completed successfully.")


if __name__ == "__main__":
	main()
