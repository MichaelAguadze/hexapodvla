"""Task definitions and management for VLA data collection."""
from pathlib import Path

import pandas as pd


# Default tasks for the hexapod navigation demo.
# Edit this list to match your setup and objects.
DEFAULT_TASKS = [
    "go to the left of the box",
    "go to the right of the box",
    "go forward to the box",
    "go behind the box",
]


class TaskManager:
    """Manages task descriptions and their integer indices."""

    def __init__(self, tasks: list[str] | None = None):
        self.tasks = list(DEFAULT_TASKS if tasks is None else tasks)
        self._task_to_index = {t: i for i, t in enumerate(self.tasks)}

    def get_task(self, index: int) -> str:
        return self.tasks[index]

    def get_index(self, task: str) -> int:
        if task not in self._task_to_index:
            self._task_to_index[task] = len(self.tasks)
            self.tasks.append(task)
        return self._task_to_index[task]

    def list_tasks(self) -> list[tuple[int, str]]:
        return list(enumerate(self.tasks))

    def print_tasks(self) -> None:
        print("\n  Available tasks:")
        for i, task in enumerate(self.tasks):
            print(f"    [{i}] {task}")
        print()

    def to_parquet(self, path: Path) -> None:
        """Write meta/tasks.parquet in LeRobot v3.0 format."""
        path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame({
            "task_index": range(len(self.tasks)),
            "task": self.tasks,
        })
        df.to_parquet(path, index=False)

    def __len__(self) -> int:
        return len(self.tasks)
