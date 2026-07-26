import os
import sys
from dulwich.repo import Repo
from dulwich.porcelain import init, add, commit, remote_add

repo_dir = r"C:\Users\Akshay.JOHN-XAVIER\OneDrive - Akkodis\Documents\Me\trade_kronos"
os.chdir(repo_dir)

print("=" * 80)
print("INITIALIZING AND COMMITTING REPOSITORY VIA DULWICH GIT ENGINE...")
print("=" * 80)

# 1. Initialize Git Repo
git_dir = os.path.join(repo_dir, ".git")
if not os.path.exists(git_dir):
    repo = init(repo_dir)
    print("Initialized new Git repository.")
else:
    repo = Repo(repo_dir)
    print("Opened existing Git repository.")

# 2. Configure Remote URL
remote_url = "https://github.com/AkshayJohn03/Kronos_Backtest.git"
try:
    remote_add(repo, "origin", remote_url)
    print(f"Added remote origin: {remote_url}")
except Exception as e:
    print(f"Remote origin note: {e}")

# 3. Create .gitignore if missing
gitignore_path = os.path.join(repo_dir, ".gitignore")
if not os.path.exists(gitignore_path):
    with open(gitignore_path, "w", encoding="utf-8") as f:
        f.write(".venv/\nvenv/\n__pycache__/\n*.pyc\n*.pyo\nkronos_temp/\n*.zip\n")

# 4. Stage all files recursively
print("Staging all project files...")
ignore_dirs = {".venv", "venv", "__pycache__", ".git", "kronos_temp"}

staged_files = []
for root, dirs, files in os.walk(repo_dir):
    dirs[:] = [d for d in dirs if d not in ignore_dirs]
    for file in files:
        if file.endswith(".pyc") or file.endswith(".zip"):
            continue
        rel_path = os.path.relpath(os.path.join(root, file), repo_dir).replace("\\", "/")
        staged_files.append(rel_path)

add(repo, staged_files)
print(f"Successfully staged {len(staged_files)} files.")

# 5. Create Commit
commit_msg = b"Initial commit: Kronos AI Nifty50 IST backtest engine, dual-mode evaluation, IST data, plots & streamlit dashboard"
commit_id = commit(repo, message=commit_msg, author=b"Akshay John <akshay@example.com>", committer=b"Akshay John <akshay@example.com>")

print(f"\nSUCCESS: Created local Git commit: {commit_id.decode('ascii')[:8]}")
print(f"Remote URL configured: {remote_url}")

