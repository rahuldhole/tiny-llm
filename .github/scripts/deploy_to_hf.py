import os
import sys
from huggingface_hub import HfApi, create_repo
from dotenv import load_dotenv

load_dotenv()

def deploy():
    token = os.getenv("HF_TOKEN")
    username = os.getenv("HF_USERNAME")
    space_name = os.getenv("HF_SPACE_NAME")
    model_name = os.getenv("HF_MODEL_NAME", f"{space_name}-adapter")

    if not all([token, username, space_name]):
        print("Error: HF_TOKEN, HF_USERNAME, and HF_SPACE_NAME must be set.")
        sys.exit(1)

    api = HfApi(token=token)

    # 1. Ensure Model Repo exists (for the fine-tuned adapter)
    model_repo_id = f"{username}/{model_name}"
    try:
        create_repo(repo_id=model_repo_id, repo_type="model", exist_ok=True)
        print(f"Model repo ensured: {model_repo_id}")
    except Exception as e:
        print(f"Error creating model repo: {e}")

    # 2. Upload Adapter Weights if they exist
    adapter_path = "outputs/qwen-fine-tuned"
    if os.path.exists(adapter_path):
        print(f"Uploading adapter from {adapter_path}...")
        api.upload_folder(
            folder_path=adapter_path,
            repo_id=model_repo_id,
            repo_type="model",
            commit_message="Update fine-tuned adapter weights"
        )
    else:
        print(f"Warning: Adapter path {adapter_path} not found. Skipping weight upload.")

    # 3. Ensure Space exists
    space_repo_id = f"{username}/{space_name}"
    try:
        create_repo(repo_id=space_repo_id, repo_type="space", space_sdk="gradio", exist_ok=True)
        print(f"Space ensured: {space_repo_id}")
    except Exception as e:
        print(f"Error creating space: {e}")

    # 4. Upload App Files to Space
    # We include app.py and requirements.txt
    print("Uploading app files to Space...")
    files_to_upload = ["app.py", "requirements.txt"]
    for file in files_to_upload:
        if os.path.exists(file):
            api.upload_file(
                path_or_fileobj=file,
                path_in_repo=file,
                repo_id=space_repo_id,
                repo_type="space",
                commit_message=f"Update {file}"
            )
        else:
            print(f"Warning: {file} not found. Skipping file upload to space.")

    print("Deployment complete!")

if __name__ == "__main__":
    deploy()
