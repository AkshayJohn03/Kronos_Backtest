import urllib.request
import zipfile
import os
import shutil

urls = [
    "https://codeload.github.com/shiyu-coder/Kronos/zip/refs/heads/main",
    "https://codeload.github.com/shiyu-coder/Kronos/zip/refs/heads/master",
    "https://github.com/shiyu-coder/Kronos/archive/refs/heads/main.zip",
    "https://github.com/shiyu-coder/Kronos/archive/refs/heads/master.zip"
]

target_dir = r"C:\Users\Akshay.JOHN-XAVIER\OneDrive - Akkodis\Documents\Me\trade_kronos"
zip_path = os.path.join(target_dir, "kronos_main.zip")
extract_path = os.path.join(target_dir, "kronos_temp")
final_repo_dir = os.path.join(target_dir, "Kronos_src")

headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}

success = False
for url in urls:
    try:
        print(f"Trying URL: {url}")
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req) as response, open(zip_path, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)
        print("Download successful!")
        success = True
        break
    except Exception as e:
        print(f"Failed {url}: {e}")

if not success:
    raise RuntimeError("Could not download Kronos repository zip from any candidate URL.")

print("Extracting zip...")
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

extracted_folders = [f for f in os.listdir(extract_path) if os.path.isdir(os.path.join(extract_path, f))]
if extracted_folders:
    extracted_folder = os.path.join(extract_path, extracted_folders[0])
    if os.path.exists(final_repo_dir):
        shutil.rmtree(final_repo_dir)
    shutil.move(extracted_folder, final_repo_dir)

if os.path.exists(zip_path):
    os.remove(zip_path)
if os.path.exists(extract_path):
    shutil.rmtree(extract_path)

print(f"Kronos source repository successfully ready at {final_repo_dir}!")

