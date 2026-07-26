import os
from dulwich.repo import Repo
from dulwich.porcelain import push

repo_dir = r"C:\Users\Akshay.JOHN-XAVIER\OneDrive - Akkodis\Documents\Me\trade_kronos"
os.chdir(repo_dir)

repo = Repo(repo_dir)
remote_url = "https://github.com/AkshayJohn03/Kronos_Backtest.git"

print("Attempting push to remote GitHub repository:", remote_url)
try:
    push(repo, remote_url, refspecs=b"refs/heads/main:refs/heads/main")
    print("SUCCESSFULLY PUSHED TO GITHUB!")
except Exception as e:
    print("\n[PUSH NOTE]")
    print(f"Details: {e}")
    print("\nTo push your commit to GitHub via VS Code or command line:")
    print("1. Open your terminal or VS Code in folder: C:\\Users\\Akshay.JOHN-XAVIER\\OneDrive - Akkodis\\Documents\\Me\\trade_kronos")
    print("2. Run: git push -u origin main")
