import os
import sys
from dulwich.repo import Repo
from dulwich.porcelain import push

repo_dir = r"C:\Users\Akshay.JOHN-XAVIER\OneDrive - Akkodis\Documents\Me\trade_kronos"
os.chdir(repo_dir)

print("=" * 80)
print("GITHUB PUSH UTILITY (ROBUST DULWICH ENGINE)")
print("=" * 80)

repo = Repo(repo_dir)

# Get token from arguments
token = sys.argv[1] if len(sys.argv) > 1 else input("Enter your GitHub Personal Access Token (PAT): ").strip()
if not token:
    print("Error: No GitHub token provided.")
    sys.exit(1)

# Format authenticated URL
remote_url = f"https://AkshayJohn03:{token}@github.com/AkshayJohn03/Kronos_Backtest.git"

# Detect current HEAD reference
try:
    head_ref = repo.refs.read_ref(b"HEAD")
    if head_ref.startswith(b"ref: "):
        current_branch = head_ref[5:].strip()
    else:
        current_branch = b"refs/heads/master"
except Exception:
    current_branch = b"refs/heads/master"

print(f"Local branch detected: {current_branch.decode('utf-8')}")
print(f"Target remote repo: https://github.com/AkshayJohn03/Kronos_Backtest.git")

# Refspec: local branch -> remote main
refspec = current_branch + b":refs/heads/main"
print(f"Pushing refspec: {refspec.decode('utf-8')} ...")

try:
    push(repo, remote_url, refspecs=[refspec])
    print("\n🎉 SUCCESS! All files, IST data, plots, reports, and code successfully pushed to GitHub!")
    print("View your remote repo at: https://github.com/AkshayJohn03/Kronos_Backtest")
except Exception as e:
    # Try fallback refspec master -> master if main fails
    try:
        push(repo, remote_url, refspecs=[current_branch + b":" + current_branch])
        print("\n🎉 SUCCESS! All files successfully pushed to GitHub!")
        print("View your remote repo at: https://github.com/AkshayJohn03/Kronos_Backtest")
    except Exception as ex:
        print(f"\nPush failed: {ex}")
        print("Details:", e)
