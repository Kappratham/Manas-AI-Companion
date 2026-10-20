"""
Step 5 — Upload train.jsonl / eval.jsonl to Hugging Face Hub.
"""

import json
from pathlib import Path

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn
    console = Console()
except ImportError:
    class _Dummy:
        def print(self, *a, **k): print(a)
        def print(self, *a): print(a)
    console = _Dummy()


def load_jsonl(path: str) -> list:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def upload_to_hub(
    train_path: str,
    eval_path: str,
    repo_id: str,
    token: str,
    private: bool = True,
) -> str:
    """
    Convert train/eval JSONL to HF Dataset, push to hub as private repo.
    Returns the HF dataset URL.
    """
    from huggingface_hub import HfApi, login
    from datasets import Dataset, DatasetDict

    console.print(f"[cyan]Uploading to: https://huggingface.co/datasets/{repo_id}[/cyan]")

    login(token=token)
    api = HfApi()
    user_info = api.whoami()
    console.print(f"[green]Authenticated as: {user_info.get('name')}[/green]")

    # Create repo if needed
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)

    # Load and convert
    train_data = load_jsonl(train_path)
    eval_data  = load_jsonl(eval_path)

    console.print(f"  Train: {len(train_data)} | Eval: {len(eval_data)}")

    # Build DatasetDict
    dataset_dict = DatasetDict({
        "train": Dataset.from_list(train_data),
        "eval":  Dataset.from_list(eval_data),
    })

    # Push to hub
    try:
        dataset_dict.push_to_hub(repo_id, token=token, private=private)
        url = f"https://huggingface.co/datasets/{repo_id}"
        console.print(Panel.fit(
            f"[bold green]Dataset uploaded successfully![/bold green]\n"
            f"URL:  [bold blue]{url}[/bold blue]",
            title="Upload Complete",
        ))
        return url
    except Exception as e:
        console.print(f"[red][ERROR] Push to hub failed: {e}[/red]")
        # Fallback: individual file uploads
        console.print("[yellow]Falling back to individual file upload...[/yellow]")
        return _upload_files_individually(api, repo_id, train_path, eval_path, token, private)


def _upload_files_individually(
    api, repo_id: str, train_path: str, eval_path: str, token: str, private: bool
) -> str:
    """Fallback if push_to_hub fails — upload train.jsonl + eval.jsonl one by one."""
    files = {"train.jsonl": train_path, "eval.jsonl": eval_path}

    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
        BarColumn(), console=console,
    ) as progress:
        task = progress.add_task("Uploading files...", total=len(files))
        for name, path in files.items():
            progress.update(task, description=f"[cyan]Uploading {name}...")
            try:
                api.upload_file(
                    path_or_fileobj=str(Path(path)),
                    path_in_repo=name,
                    repo_id=repo_id,
                    repo_type="dataset",
                )
            except Exception as e:
                console.print(f"[red]  Failed {name}: {e}[/red]")
            progress.advance(task)

    url = f"https://huggingface.co/datasets/{repo_id}"
    console.print(Panel.fit(f"Uploaded. URL: [blue]{url}[/blue]", title="Done"))
    return url