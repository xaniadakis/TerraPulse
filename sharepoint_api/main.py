import os
import requests
from dotenv import load_dotenv

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
    url = f"https://graph.microsoft.com/v1.0/drives/{drive_id}/items/{parent_id}/children"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    data = {
        "name": folder_name,
        "folder": {},
        "@microsoft.graph.conflictBehavior": "rename"
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
        with open(file_path, "rb") as f:
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
                    break
                response.raise_for_status()
                start += len(chunk)

def sync_directory(local_path, parent_id, drive_id, token):
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
            print(f"Uploading: {file_path}")
            upload_file_conditional(drive_id, current_parent_id, file_path, file_name, token)

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
    print("✅ Done.")
