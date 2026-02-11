import os
import sys
import argparse
from huggingface_hub import HfApi, create_repo
from dotenv import load_dotenv

load_dotenv()


def deploy(mode="all"):
    token = os.getenv("HF_TOKEN")
    username = os.getenv("HF_USERNAME")
    space_name = os.getenv("HF_SPACE_NAME")
    model_name = os.getenv("HF_MODEL_NAME", f"{space_name}-adapter")

    if not all([token, username, space_name]):
        print("Error: HF_TOKEN, HF_USERNAME, and HF_SPACE_NAME must be set.")
        sys.exit(1)

    api = HfApi(token=token)
    model_repo_id = f"{username}/{model_name}"
    space_repo_id = f"{username}/{space_name}"

    # ── Model Upload ──
    if mode in ("all", "model"):
        print(f"\n📦 Model repo: {model_repo_id}")
        create_repo(repo_id=model_repo_id, repo_type="model", exist_ok=True, token=token)

        adapter_path = "outputs/qwen-fine-tuned"
        if os.path.exists(adapter_path):
            print(f"   Uploading adapter from {adapter_path}...")
            api.upload_folder(
                folder_path=adapter_path,
                repo_id=model_repo_id,
                repo_type="model",
                commit_message="Update fine-tuned adapter weights",
            )
            print("   ✅ Model uploaded!")
        else:
            print(f"   ❌ Adapter path '{adapter_path}' not found. Run 'task train' first.")
            sys.exit(1)

    # ── Space Upload ──
    if mode in ("all", "space"):
        print(f"\n🚀 Space: {space_repo_id}")
        create_repo(repo_id=space_repo_id, repo_type="space", space_sdk="gradio", exist_ok=True, token=token)

        # Upload app.py
        if os.path.exists("app.py"):
            api.upload_file(
                path_or_fileobj="app.py",
                path_in_repo="app.py",
                repo_id=space_repo_id,
                repo_type="space",
                commit_message="Update app.py",
            )

        # Upload requirements-space.txt as requirements.txt in the Space
        req_file = "requirements-space.txt"
        if os.path.exists(req_file):
            api.upload_file(
                path_or_fileobj=req_file,
                path_in_repo="requirements.txt",
                repo_id=space_repo_id,
                repo_type="space",
                commit_message="Update requirements.txt",
            )

        print("   ✅ Space updated!")

    print("\n🎉 Deployment complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["all", "model", "space"], default="all",
                        help="What to deploy: 'model', 'space', or 'all'")
    args = parser.parse_args()
    deploy(args.mode)
