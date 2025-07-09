# SharePoint Directory Uploader

This script recursively uploads a local folder (with all subfolders and files) to a SharePoint drive using the Microsoft Graph API.

---

We automatically fetch access token via OAuth2 w/ client credentials flow.
Then we recursively create folders and upload files to SharePoint w/ support for large files via upload sessions.

## 📊 Requirements

Azure AD App registration (now part of Microsoft Entra) with:
* `client_id`
* `client_secret`
* `Sites.ReadWrite.All` or `Files.ReadWrite.All` permissions (Application)

> ⚠️ You must have an application registered in [Microsoft Entra ID](https://entra.microsoft.com/) (formerly Azure Active Directory), 
generate a secret, and grant it API permissions to Microsoft Graph.

---

## 📁 .env Configuration

Create a `.env` file in the root directory with the following variables:

```env
CLIENT_ID=your-client-id
CLIENT_SECRET=your-client-secret
TENANT_ID=your-tenant-id
SITE_ID=your-site-id-or-domain,name   # e.g. formula1.sharepoint.com,site-name
LOCAL_DIR=./path/to/local/folder      # Local folder to upload
```

---

## 🚀 How to Run

1. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

2. **Add your credentials** to the `.env` file

3. **Run the script**:

   ```bash
   python main.py
   ```

---

## ⚡ Notes

* Uploads large files in 10MB chunks
* Folder structure is preserved in SharePoint
* All file uploads use `@microsoft.graph.conflictBehavior=rename` to avoid overwrites

---

## 🌐 Microsoft Docs

* [Create Upload Session](https://learn.microsoft.com/en-us/graph/api/driveitem-createuploadsession)
* [Access Token (OAuth2)](https://learn.microsoft.com/en-us/azure/active-directory/develop/v2-oauth2-client-creds-grant-flow)
* [Register an app in Microsoft Entra ID](https://learn.microsoft.com/en-us/entra/identity-platform/app-registration-overview)
