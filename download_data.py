import os
import re
import gdown

SAVE_DIR = "./dataset/downloaded_data"
GDRIVE_URL = "https://drive.google.com/drive/folders/163rOt9U4sfyoCxpPYRA-PZt_1VJ6jmUw?usp=sharing"

def download_drive_folder(gdrive_url: str, save_dir: str):
    """
    Download all files inside a public Google Drive folder using gdown.

    Args:
        gdrive_url (str): Link của Google Drive folder
        save_dir (str): Thư mục bạn muốn lưu file

    Returns:
        None
    """
    os.makedirs(save_dir, exist_ok=True)

    match = re.search(r"folders\/([a-zA-Z0-9_-]+)", gdrive_url)
    if not match:
        raise ValueError("Không tìm thấy folder ID trong URL.")

    folder_id = match.group(1)

    gdown_url = f"https://drive.google.com/drive/folders/{folder_id}?usp=sharing"

    print(f"🔽 Đang tải toàn bộ file từ folder: {folder_id}")
    print(f"📁 Lưu vào: {save_dir}")

    gdown.download_folder(
        url=gdown_url,
        output=save_dir,
        quiet=False,
        use_cookies=False,
    )

    print("✅ Tải xong toàn bộ file!")


if __name__ == "__main__":
    download_drive_folder(
        gdrive_url=GDRIVE_URL,
        save_dir=SAVE_DIR
    )
