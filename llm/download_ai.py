from huggingface_hub import snapshot_download

def manual_download(model_id='MediaTek-Research/Llama-Breeze2-3B-Instruct-v0_1', local_dir='./local_breeze2_3b'):
    print(f"開始下載模型 {model_id} 到本地資料夾 {local_dir} ...")
    snapshot_download(repo_id=model_id, local_dir=local_dir, local_dir_use_symlinks=False)
    print("下載完成！")

if __name__ == '__main__':
    manual_download()
