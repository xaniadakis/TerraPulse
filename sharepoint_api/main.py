import os
import requests
from dotenv import load_dotenv
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

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
            tqdm.write(f"📁 Folder exists: {folder_name}")
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

def validate_item(args):
    item_type, rel_path, drive_id, token, remote_parent_id = args
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
            return f"Missing folder: {os.path.join(*parent_parts)}"
        current_id = match["id"]

    if current_id is None:
        return f"Missing path: {rel_path}"

    url = f"https://graph.microsoft.com/v1.0/drives/{drive_id}/items/{current_id}/children"
    headers = {"Authorization": f"Bearer {token}"}
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    children = resp.json().get("value", [])

    if item_type == "folder":
        found = any(child["name"] == item_name and "folder" in child for child in children)
        if not found:
            return f"Missing folder: {rel_path}"
    else:
        found = any(child["name"] == item_name and "file" in child for child in children)
        if not found:
            return f"Missing file: {rel_path}"
    return None

def validate_upload(drive_id, token, local_path, remote_parent_id="root"):
    print("🔍 Validating upload...")

    local_items = []
    for root, dirs, files in os.walk(local_path):
        rel_path = os.path.relpath(root, local_path)
        for d in dirs:
            local_items.append(("folder", os.path.normpath(os.path.join(rel_path, d))))
        for f in files:
            local_items.append(("file", os.path.normpath(os.path.join(rel_path, f))))

    args_list = [(item_type, rel_path, drive_id, token, remote_parent_id) for item_type, rel_path in local_items]

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(tqdm(executor.map(validate_item, args_list), total=len(args_list), desc="Validating items"))

    mismatches = [r for r in results if r]
    if mismatches:
        print("❌ Validation failed. Missing items:")
        for m in mismatches:
            print("-", m)
    else:
        print("✅ Validation successful. All files and folders are present.")

def load_ignore_lists(ignore_file=".uploadignore"):
    ignored_dirs = set()
    ignored_files = set()
    if os.path.exists(ignore_file):
        with open(ignore_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip().strip("\ufeff")
                if not line or line.startswith("#"):
                    continue

                # Use path relative to LOCAL_DIR if possible
                local_path = os.path.join(LOCAL_DIR, line)
                if os.path.isdir(local_path):
                    ignored_dirs.add(line)
                elif os.path.isfile(local_path):
                    ignored_files.add(line)
                else:
                    # Default: assume it's a dir if not present
                    ignored_dirs.add(line)
    return ignored_dirs, ignored_files

def sync_directory(local_path, parent_id, drive_id, token):
    ignored_dirs, ignored_files = load_ignore_lists()
    print("🔽 Ignoring dirs:", ignored_dirs)
    print("🔽 Ignoring files:", ignored_files)
    all_files = []
    folder_map = {}

    # Collect eligible files
    for root, dirs, files in os.walk(local_path):
        dirs[:] = [d for d in dirs if d not in ignored_dirs]

        rel_path = os.path.relpath(root, local_path)
        folder_map[root] = rel_path.split(os.sep) if rel_path != '.' else []

        for f in files:
            if f in ignored_files:
                continue
            all_files.append((root, f))

    # Print directories that will be uploaded (not ignored)
    top_level_dirs = []
    for d in os.listdir(local_path):
        full_path = os.path.join(local_path, d)
        if not os.path.isdir(full_path):
            continue
        d_clean = d.strip().strip("\ufeff")
        if d_clean in ignored_dirs:
            continue
        top_level_dirs.append(d)

    # Upload
    with tqdm(total=len(all_files), desc="Uploading files") as file_bar:
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for root, file_name in all_files:
                cloud_path_parts = folder_map[root]
                current_parent_id = parent_id

                for part in cloud_path_parts:
                    if part not in folder_cache.get(current_parent_id, {}):
                        new_folder_id = create_folder(drive_id, current_parent_id, part, token)
                        folder_cache.setdefault(current_parent_id, {})[part] = new_folder_id
                    current_parent_id = folder_cache[current_parent_id][part]

                file_path = os.path.join(root, file_name)
                futures.append(executor.submit(upload_and_tick, drive_id, current_parent_id, file_path, file_name, token, file_bar))

            for f in futures:
                f.result()

def upload_and_tick(drive_id, parent_id, file_path, file_name, token, file_bar):
    upload_file_conditional(drive_id, parent_id, file_path, file_name, token)
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