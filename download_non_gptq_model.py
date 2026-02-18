import os
from huggingface_hub import snapshot_download

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
TARGET_DIR = "./models2"


def main():
    os.makedirs(TARGET_DIR, exist_ok=True)

    print(f"Downloading {MODEL_ID} into {TARGET_DIR} ...")
    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=TARGET_DIR,
        local_dir_use_symlinks=False,
        revision="main",
    )
    print("Done.")
    print("Now run: python chatbot_smart-nonllamacpp.py")


if __name__ == "__main__":
    main()
