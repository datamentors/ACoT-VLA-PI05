
import collections
import json
import logging
import pathlib
from collections.abc import Collection, Mapping, Sequence

import numpy as np
import torch
import random
from tqdm import tqdm
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset

def get_base_dataset(ds):
    if hasattr(ds, "_dataset"):
        return get_base_dataset(ds._dataset)
    return ds

def sample_subtask(dataset):
    valid_intervals = []
    base_ds = get_base_dataset(dataset)
    
    sub_datasets = []
    
    if isinstance(base_ds, lerobot_dataset.MultiLeRobotDataset):
        for sub_ds in base_ds._datasets:
            sub_datasets.append(sub_ds)
    else:
        sub_datasets.append(dataset)

    current_global_offset = 0
    total_episodes_processed = 0

    print(f"Processing {len(sub_datasets)} sub-datasets...")

    for sub_ds in sub_datasets:
        inner_ds = get_base_dataset(sub_ds)
        
        instruction_segment = inner_ds.meta.info.get('instruction_segments', {})
        episode_data_index = inner_ds.episode_data_index
        num_episodes = len(episode_data_index['from'])
        
        for ep_idx in range(num_episodes):
            local_episode_start = episode_data_index['from'][ep_idx].item()
            
            if str(ep_idx) not in instruction_segment:
                continue

            tasks = instruction_segment[str(ep_idx)]
            for subtask in tasks:
                local_start = subtask["start_frame_index"] + local_episode_start
                end_frame_key = "success_frame_index" if "success_frame_index" in subtask else "end_frame_index"
                local_end = subtask[end_frame_key] + local_episode_start
                
                instruction = subtask["instruction"].lower()
                is_reset = any(k in instruction for k in ['reset', 'return', 'default'])
                
                if is_reset:
                    if local_end - local_start > 90:
                        local_end = local_start + 45
                
                global_start = local_start + current_global_offset
                global_end = local_end + current_global_offset
                
                valid_intervals.append((global_start, global_end))
        
        current_global_offset += len(sub_ds)
        total_episodes_processed += num_episodes

    print(f"Total {len(valid_intervals)} valid intervals from {total_episodes_processed} episodes.")
    return valid_intervals


def session_key(root) -> str:
    """Key of a LeRobot dataset in an episode allow-list: its last two path parts.

    Args:
        root: Dataset root, e.g. ``.../g2a0004-infpose-headset_loop/2026-09-24/s01``.

    Returns:
        ``"<parent>/<name>"``, e.g. ``"2026-09-24/s01"``.
    """
    path = pathlib.Path(root)
    return f"{path.parent.name}/{path.name}"


def load_episode_include(path: str | None) -> dict[str, frozenset[int]] | None:
    """Load a per-dataset episode allow-list.

    Args:
        path: JSON file mapping ``session_key`` -> list of LeRobot ``episode_index`` values to keep.
            Keys starting with ``_`` are comments. ``None`` means no episode filter.

    Returns:
        Mapping of session key to allowed episode indices, or ``None`` when ``path`` is ``None``.

    Raises:
        ValueError: If the file is not a JSON object of integer lists.
    """
    if path is None:
        return None
    raw = json.loads(pathlib.Path(path).read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a JSON object of session -> episode list")
    allow: dict[str, frozenset[int]] = {}
    for key, episodes in raw.items():
        if key.startswith("_"):
            continue
        if not isinstance(episodes, list) or not all(isinstance(e, int) and not isinstance(e, bool) for e in episodes):
            raise ValueError(f"{path}: {key} must map to a list of integer episode indices")
        allow[key] = frozenset(episodes)
    return allow


def sample_tasks(
    dataset,
    task_include: Sequence[str],
    episode_include: Mapping[str, Collection[int]] | None = None,
    *,
    require_all_episode_keys: bool = False,
) -> list[int]:
    """Global indices of the frames whose LeRobot task text is one of task_include.

    For LeRobot v2.1 exports that write each sub-task as its own task (e.g. LOOP): the per-frame
    task_index is then the sub-task label. Matched by text, not index, because every dataset in a
    MultiLeRobot list numbers its tasks.jsonl independently. Global indices follow the same
    child-after-child layout as TransformedDataset.

    Args:
        dataset: LeRobot dataset, MultiLeRobotDataset, or a wrapper of either.
        task_include: Task texts to keep.
        episode_include: Optional ``session_key`` -> allowed ``episode_index`` values. Datasets whose
            key is listed keep only those episodes; unlisted datasets keep all episodes.
        require_all_episode_keys: Raise if an ``episode_include`` key matches no dataset (training
            sets this to catch typos; validation runs on other repos and leaves it off).

    Returns:
        Sorted-by-dataset global frame indices that pass both filters.

    Raises:
        ValueError: On row-count mismatch, unmatched task texts, allowed episodes absent from a
            dataset, or (with ``require_all_episode_keys``) unmatched allow-list keys.
    """
    wanted = {t.strip() for t in task_include}
    base_ds = get_base_dataset(dataset)
    if isinstance(base_ds, lerobot_dataset.MultiLeRobotDataset):
        sub_datasets = list(base_ds._datasets)
    else:
        sub_datasets = [dataset]

    indices: list[int] = []
    frames_per_task: collections.Counter = collections.Counter()
    matched_keys: set[str] = set()
    offset = 0
    for sub_ds in sub_datasets:
        inner_ds = get_base_dataset(sub_ds)
        hits = {i: t.strip() for i, t in inner_ds.meta.tasks.items() if t.strip() in wanted}
        key = session_key(inner_ds.root)
        allowed = None if episode_include is None else episode_include.get(key)
        if allowed is not None:
            matched_keys.add(key)
        if hits:
            task_index = inner_ds.hf_dataset.data.column("task_index").to_numpy().reshape(-1)
            if len(task_index) != len(sub_ds):
                raise ValueError(f"{inner_ds.root}: task_index has {len(task_index)} rows, dataset has {len(sub_ds)}")
            keep = np.isin(task_index, list(hits))
            if allowed is not None:
                episode_index = inner_ds.hf_dataset.data.column("episode_index").to_numpy().reshape(-1)
                absent = sorted(set(allowed) - set(np.unique(episode_index).tolist()))
                if absent:
                    raise ValueError(f"{inner_ds.root}: allowed episodes not in dataset: {absent}")
                keep &= np.isin(episode_index, list(allowed))
            local = np.flatnonzero(keep)
            indices.extend((local + offset).tolist())
            counts = {t: int(np.sum(task_index[keep] == i)) for i, t in hits.items()}
            frames_per_task.update(counts)
            episodes_note = "all episodes" if allowed is None else f"{len(allowed)} allowed episodes"
            logging.info(
                "subtask filter: %s -> %d / %d frames %s (%s)",
                inner_ds.root,
                len(local),
                len(sub_ds),
                counts,
                episodes_note,
            )
        else:
            logging.info("subtask filter: %s -> 0 / %d frames (no matching task)", inner_ds.root, len(sub_ds))
        offset += len(sub_ds)

    if require_all_episode_keys and episode_include is not None:
        unmatched = sorted(set(episode_include) - matched_keys)
        if unmatched:
            raise ValueError(f"episode_include keys matched no dataset: {unmatched}")
    missing = sorted(wanted - {t for t, n in frames_per_task.items() if n > 0})
    if missing:
        raise ValueError(f"subtask_include entries matched no frames in any dataset: {missing}")
    logging.info("subtask filter total: %d / %d frames %s", len(indices), offset, dict(frames_per_task))
    return indices


class FrameSampler(torch.utils.data.Sampler):
    """
    Custom sampler that only samples data indices falling within specified intervals
    """
    def __init__(
        self,
        dataset,
        sampler_type,
        seed: int = 0,
        task_include: Sequence[str] = (),
        episode_include: Mapping[str, Collection[int]] | None = None,
    ):
        self.task_include = tuple(task_include)
        self.episode_include = episode_include
        valid_intervals = self.parse_dataset(dataset, sampler_type)
        self.sample_frames(valid_intervals, len(dataset), seed=seed)

    def parse_dataset(self, dataset, sampler_type):
        """
        Args:
            intervals: List of (start_index, end_index) tuples
        """
        if sampler_type == 'subtask':
            return sample_subtask(dataset)
        elif sampler_type == 'task':
            frames = sample_tasks(
                dataset, self.task_include, self.episode_include, require_all_episode_keys=True
            )
            return [(i, i) for i in frames]
        else:
            raise ValueError(f"Invalid sampler type: {sampler_type}")

    def sample_frames(self, intervals, dataset_size, seed: int = 0):
        """
        Args:
            intervals: List of (start_index, end_index) tuples
            dataset_size: Total size of the dataset
            seed: RNG seed used to shuffle the valid indices deterministically.
        """
        self.intervals = intervals
        self.dataset_size = dataset_size

        # Pre-compute all valid indices
        self.valid_indices = []
        for start_idx, end_idx in intervals:
            # Ensure indices are within dataset bounds
            start_idx = max(0, start_idx)
            end_idx = min(dataset_size - 1, end_idx)

            # Add all indices within the interval
            self.valid_indices.extend(range(start_idx, end_idx + 1))

        # Remove duplicates and sort
        self.valid_indices = sorted(list(set(self.valid_indices)))
        print(f"Total {len(self.valid_indices)} valid indices,", 'original:', dataset_size)

        random.Random(seed).shuffle(self.valid_indices)
    
    def __iter__(self):
        return iter(self.valid_indices)
    
    def __len__(self):
        return len(self.valid_indices)
