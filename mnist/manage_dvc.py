from dvc.repo import Repo
from pathlib import Path

def dvc_push(list_data_path):
    repo = Repo(".")
    for path in list_data_path:
        if Path(path).exists():
            repo.add(path)
        else:
            print(f'{path} does not exist.')
    repo.commit()
    repo.push()


def dvc_pull():
    repo = Repo(".")
    repo.pull(force=True)