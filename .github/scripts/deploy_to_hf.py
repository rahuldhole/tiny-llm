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
        print("❌ HF_TOKEN, HF_USERNAME, and HF_SPACE_NAME must be set.")
        sys.exit(1)

    api = HfApi(token=token)
    model_repo_id = f"{username}/{model_name}"
    space_repo_id = f"{username}/{space_name}"

    # ── Model Upload ──
    if mode in ("all", "model"):
        print(f"\n📦 Model: {model_repo_id}")
        create_repo(repo_id=model_repo_id, repo_type="model", exist_ok=True, token=token)

        adapter_path = "outputs/qwen-fine-tuned"
        if not os.path.exists(adapter_path):
            print(f"   ❌ '{adapter_path}' not found. Run 'task train' first.")
            sys.exit(1)

        print("   Uploading adapter...")
        api.upload_folder(
            folder_path=adapter_path,
            repo_id=model_repo_id,
            repo_type="model",
            commit_message="Update fine-tuned adapter weights",
        )

        # Upload training config as metadata
        config_file = "configs/train_config.yaml"
        if os.path.exists(config_file):
            api.upload_file(
                path_or_fileobj=config_file,
                path_in_repo="train_config.yaml",
                repo_id=model_repo_id,
                repo_type="model",
                commit_message="Update training config",
            )
        print("   ✅ Model uploaded!")

    # ── Space Upload ──
    if mode in ("all", "space"):
        print(f"\n🚀 Space: {space_repo_id}")
        create_repo(
            repo_id=space_repo_id, repo_type="space",
            space_sdk="gradio", exist_ok=True, token=token
        )

        for local, remote in [
            ("app.py", "app.py"),
            ("requirements-space.txt", "requirements.txt"),
        ]:
            if os.path.exists(local):
                api.upload_file(
                    path_or_fileobj=local,
                    path_in_repo=remote,
                    repo_id=space_repo_id,
                    repo_type="space",
                    commit_message=f"Update {remote}",
                )
        print("   ✅ Space updated!")

    print("\n🎉 Deployment complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deploy Tiny LLM to Hugging Face")
    parser.add_argument("--mode", choices=["all", "model", "space"], default="all")
    args = parser.parse_args()
    deploy(args.mode)
