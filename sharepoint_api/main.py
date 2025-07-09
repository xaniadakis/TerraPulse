import os
import requests
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
TENANT_ID = os.getenv("TENANT_ID")
SITE_ID = os.getenv("SITE_ID")
LOCAL_DIR = os.getenv("LOCAL_DIR")

def get_access_token():
    url = f"https://login.microsoftonline.com/{TENANT_ID}/oauth2/v2.0/token"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    data = {
        "client_id": CLIENT_ID,
        "scope": "https://graph.microsoft.com/.default",
        "client_secret": CLIENT_SECRET,
        "grant_type": "client_credentials"
    }
    resp = requests.post(url, headers=headers, data=data)
    resp.raise_for_status()
    return resp.json()["access_token"]

def get_drive_id(token):
    url = f"https://graph.microsoft.com/v1.0/sites/{SITE_ID}/drives"
    headers = {"Authorization": f"Bearer {token}"}
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    return resp.json()["value"][0]["id"]

def create_folder(drive_id, parent_id, folder_name, token):
    # First, check if folder already exists
    url = f"https://graph.microsoft.com/v1.0/drives/{drive_id}/items/{parent_id}/children"
    headers = {"Authorization": f"Bearer {token}"}
    params = {"$top": 999}  # in case the parent has many children
    resp = requests.get(url, headers=headers, params=params)
    resp.raise_for_status()
    children = resp.json().get("value", [])

    for child in children:
        if child["name"] == folder_name and "folder" in child:
            print(f"📁 Folder exists: {folder_name}")
            return child["id"]  # Reuse existing folder

    # Folder doesn't exist, create it
    url = f"https://graph.microsoft.com/v1.0/drives/{drive_id}/items/{parent_id}/children"
    headers["Content-Type"] = "application/json"
    data = {
        "name": folder_name,
        "folder": {}
    }
    resp = requests.post(url, headers=headers, json=data)
    resp.raise_for_status()
    return resp.json()["id"]

def create_upload_session(drive_id, parent_id, file_name, token):
    url = f"https://graph.microsoft.com/v1.0/drives/{drive_id}/items/{parent_id}:/{file_name}:/createUploadSession"
    headers = {
        "Authorization": f"Bearer {token}"
    }
    resp = requests.post(url, headers=headers)
    resp.raise_for_status()
    return resp.json()["uploadUrl"]

def upload_file_conditional(drive_id, parent_id, file_path, file_name, token):
    file_size = os.path.getsize(file_path)

    if file_size < 4 * 1024 * 1024:
        # Small file: use PUT
        url = f"https://graph.microsoft.com/v1.0/drives/{drive_id}/items/{parent_id}:/{file_name}:/content"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/octet-stream"
        }
        with open(file_path, "rb") as f:
            resp = requests.put(url, headers=headers, data=f)
            resp.raise_for_status()
    else:
        # Large file: use upload session
        upload_url = create_upload_session(drive_id, parent_id, file_name, token)
        chunk_size = 10 * 1024 * 1024
        with open(file_path, "rb") as f, tqdm(total=file_size, unit='B', unit_scale=True, desc=file_name, leave=False) as pbar:
            start = 0
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                end = start + len(chunk) - 1
                headers = {
                    "Content-Length": str(len(chunk)),
                    "Content-Range": f"bytes {start}-{end}/{file_size}"
                }
                response = requests.put(upload_url, headers=headers, data=chunk)
                if response.status_code in [200, 201]:
                    pbar.update(len(chunk))
                    break
                response.raise_for_status()
                pbar.update(len(chunk))
                start += len(chunk)

from tqdm import tqdm

def validate_upload(drive_id, token, local_path, remote_parent_id="root"):
    print("🔍 Validating upload...")

    # Pre-count all files and folders
    local_items = []
    for root, dirs, files in os.walk(local_path):
        rel_path = os.path.relpath(root, local_path)
        for d in dirs:
            local_items.append(("folder", os.path.normpath(os.path.join(rel_path, d))))
        for f in files:
            local_items.append(("file", os.path.normpath(os.path.join(rel_path, f))))

    mismatches = []

    for item_type, rel_path in tqdm(local_items, desc="Validating items"):
        path_parts = rel_path.split(os.sep) if rel_path != "." else []
        parent_parts = path_parts[:-1]
        item_name = path_parts[-1]
        current_id = remote_parent_id

        # Navigate to parent folder
        for part in parent_parts:
            url = f"https://graph.microsoft.com/v1.0/drives/{drive_id}/items/{current_id}/children"
            headers = {"Authorization": f"Bearer {token}"}
            resp = requests.get(url, headers=headers)
            resp.raise_for_status()
            children = resp.json().get("value", [])
            match = next((item for item in children if item["name"] == part and "folder" in item), None)
            if not match:
                mismatches.append(f"Missing folder: {os.path.join(*parent_parts)}")
                current_id = None
                break
            current_id = match["id"]

        if current_id is None:
            continue

        # Check if item exists
        url = f"https://graph.microsoft.com/v1.0/drives/{drive_id}/items/{current_id}/children"
        headers = {"Authorization": f"Bearer {token}"}
        resp = requests.get(url, headers=headers)
        resp.raise_for_status()
        children = resp.json().get("value", [])

        if item_type == "folder":
            found = any(child["name"] == item_name and "folder" in child for child in children)
            if not found:
                mismatches.append(f"Missing folder: {rel_path}")
        else:
            found = any(child["name"] == item_name and "file" in child for child in children)
            if not found:
                mismatches.append(f"Missing file: {rel_path}")

    if mismatches:
        print("❌ Validation failed. Missing items:")
        for item in mismatches:
            print("-", item)
    else:
        print("✅ Validation successful. All files and folders are present.")

def sync_directory(local_path, parent_id, drive_id, token):
    all_files = []
    for root, _, files in os.walk(local_path):
        for f in files:
            all_files.append(os.path.join(root, f))

    with tqdm(total=len(all_files), desc="Uploading files") as file_bar:
        for root, dirs, files in os.walk(local_path):
            rel_path = os.path.relpath(root, local_path)
            cloud_path_parts = rel_path.split(os.sep) if rel_path != '.' else []

            current_parent_id = parent_id

            for part in cloud_path_parts:
                if part not in folder_cache.get(current_parent_id, {}):
                    new_folder_id = create_folder(drive_id, current_parent_id, part, token)
                    folder_cache.setdefault(current_parent_id, {})[part] = new_folder_id
                current_parent_id = folder_cache[current_parent_id][part]

            for file_name in files:
                file_path = os.path.join(root, file_name)
                upload_file_conditional(drive_id, current_parent_id, file_path, file_name, token)
                file_bar.update(1)


if __name__ == "__main__":
    print("Fetching token...")
    token = get_access_token()
    print("Getting drive ID...")
    drive_id = get_drive_id(token)
    print(f"Drive ID: {drive_id}")
    global folder_cache
    folder_cache = {}
    print("Syncing directory...")
    sync_directory(LOCAL_DIR, "root", drive_id, token)
    print("✅ Done Uploading, shall now validate...")
    validate_upload(drive_id, token, LOCAL_DIR)

