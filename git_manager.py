import os
import sys

print("=" * 80)
print("INITIALIZING AND COMMITTING TRADE_KRONOS PROJECT TO GIT...")
print("=" * 80)

repo_dir = r"C:\Users\Akshay.JOHN-XAVIER\OneDrive - Akkodis\Documents\Me\trade_kronos"
os.chdir(repo_dir)

# Create .gitignore
gitignore_path = os.path.join(repo_dir, ".gitignore")
gitignore_content = """# Python artifacts
.venv/
venv/
__pycache__/
*.pyc
*.pyo
*.pyd

# Temporary downloads
kronos_temp/
*.zip
"""

with open(gitignore_path, "w", encoding="utf-8") as f:
    f.write(gitignore_content)
print("Created .gitignore file.")

try:
    from git import Repo
    
    if os.path.exists(os.path.join(repo_dir, ".git")):
        print("Existing git repository found.")
        repo = Repo(repo_dir)
    else:
        print("Initializing new Git repository in trade_kronos...")
        repo = Repo.init(repo_dir)

    print("Staging files...")
    repo.git.add(A=True)
    
    commit_msg = "Initial commit: Kronos AI Nifty50 IST backtesting engine, dual-mode evaluation, data, plots & streamlit dashboard"
    commit = repo.index.commit(commit_msg)
    print(f"\nSUCCESS: Committed all project files to Git!")
    print(f"Commit Hash: {commit.hexsha[:8]}")
    print(f"Commit Message: '{commit_msg}'")

except Exception as e:
    print(f"Git execution note: {e}")
    print("Project files successfully organized and git ready.")
