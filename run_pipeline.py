import subprocess
import sys


def run_stage(script):
    print(f"\nRunning {script}...")
    result = subprocess.run([sys.executable, script], check=False)
    if result.returncode != 0:
        raise RuntimeError(f"{script} failed with exit code {result.returncode}")


if __name__ == "__main__":
    run_stage("preprocess.py")
    run_stage("train.py")
    print("\nPipeline completed successfully")
