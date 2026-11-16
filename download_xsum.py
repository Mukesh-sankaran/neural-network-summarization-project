from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="EdinburghNLP/xsum",
    repo_type="dataset",
    local_dir="./datasets/xsum"
)

print("XSum downloaded successfully!")