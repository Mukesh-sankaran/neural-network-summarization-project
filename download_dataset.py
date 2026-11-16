from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="abisee/cnn_dailymail",
    repo_type="dataset",
    allow_patterns="3.0.0/*",
    local_dir="./datasets/cnn_dailymail"
)

print("Dataset downloaded successfully!")