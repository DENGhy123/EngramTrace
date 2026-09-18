from pathlib import Path

import h5py
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = (
    PROJECT_ROOT
    / "data"
    / "libero"
    / "libero_spatial"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def demo_sort_key(name):
    return int(name.split("_")[1])


def binary_gripper_state(gripper):
    """
    Convert gripper command into binary state.

    LIBERO / robosuite convention:
        -1 -> open
        +1 -> closed

    We use > 0 as closed.
    """
    return (gripper > 0).astype(np.int32)


def find_closed_intervals(gripper):
    """
    Return all contiguous closed intervals.

    Each item:
        (start, end, length)

    start/end are inclusive.
    """

    state = binary_gripper_state(gripper)

    intervals = []

    start = None

    for t, closed in enumerate(state):

        if closed == 1 and start is None:
            start = t

        if closed == 0 and start is not None:
            end = t - 1

            intervals.append(
                (
                    start,
                    end,
                    end - start + 1
                )
            )

            start = None

    # Closed until trajectory ends
    if start is not None:

        end = len(state) - 1

        intervals.append(
            (
                start,
                end,
                end - start + 1
            )
        )

    return intervals


rows = []
interval_rows = []


files = sorted(DATA_ROOT.glob("*.hdf5"))

print(f"Found {len(files)} task files.")


for file_path in files:

    with h5py.File(file_path, "r") as f:

        demos = sorted(
            f["data"].keys(),
            key=demo_sort_key
        )

        for demo_name in demos:

            demo = f["data"][demo_name]

            actions = np.asarray(
                demo["actions"]
            )

            ee_pos = np.asarray(
                demo["obs"]["ee_pos"]
            )

            gripper = actions[:, 6]

            T = len(gripper)

            intervals = find_closed_intervals(
                gripper
            )

            # Record all intervals
            for idx, (start, end, length) in enumerate(
                intervals
            ):

                interval_rows.append(
                    {
                        "task_file": file_path.name,
                        "demo": demo_name,
                        "interval_id": idx,
                        "start": start,
                        "end": end,
                        "length": length,
                        "start_fraction": start / T,
                        "end_fraction": end / T,
                    }
                )

            if intervals:

                # Main holding interval = longest closed interval
                main_interval = max(
                    intervals,
                    key=lambda x: x[2]
                )

                main_grasp = main_interval[0]
                main_closed_end = main_interval[1]
                main_holding_steps = main_interval[2]

                # Release = first open step after main closed interval
                if main_closed_end < T - 1:
                    main_release = main_closed_end + 1
                else:
                    main_release = None

                grasp_fraction = (
                    main_grasp / T
                )

                release_fraction = (
                    main_release / T
                    if main_release is not None
                    else np.nan
                )

            else:

                main_grasp = None
                main_release = None
                main_holding_steps = 0
                grasp_fraction = np.nan
                release_fraction = np.nan

            # Path length
            delta_pos = np.diff(
                ee_pos,
                axis=0
            )

            path_length = np.linalg.norm(
                delta_pos,
                axis=1
            ).sum()

            # Ratio between longest and second longest
            lengths = sorted(
                [x[2] for x in intervals],
                reverse=True
            )

            longest = (
                lengths[0]
                if len(lengths) >= 1
                else 0
            )

            second_longest = (
                lengths[1]
                if len(lengths) >= 2
                else 0
            )

            dominance_ratio = (
                longest / second_longest
                if second_longest > 0
                else np.inf
            )

            rows.append(
                {
                    "task_file": file_path.name,
                    "demo": demo_name,
                    "num_steps": T,
                    "num_closed_intervals":
                        len(intervals),
                    "main_grasp":
                        main_grasp,
                    "main_release":
                        main_release,
                    "main_holding_steps":
                        main_holding_steps,
                    "grasp_fraction":
                        grasp_fraction,
                    "release_fraction":
                        release_fraction,
                    "dominance_ratio":
                        dominance_ratio,
                    "ee_path_length":
                        path_length,
                }
            )


df = pd.DataFrame(rows)
interval_df = pd.DataFrame(interval_rows)


summary_path = (
    OUTPUT_DIR
    / "libero_main_grasp_statistics.csv"
)

interval_path = (
    OUTPUT_DIR
    / "libero_all_closed_intervals.csv"
)


df.to_csv(
    summary_path,
    index=False
)

interval_df.to_csv(
    interval_path,
    index=False
)


print("\nSaved:")
print(summary_path)
print(interval_path)


print("\n" + "=" * 72)
print("Main-grasp interval summary")
print("=" * 72)


print("\nNumber of demonstrations:")
print(len(df))


print("\nNumber of closed intervals per demo:")
print(
    df["num_closed_intervals"]
    .value_counts()
    .sort_index()
)


print("\nMain grasp fraction:")
print(
    df["grasp_fraction"].describe()
)


print("\nMain release fraction:")
print(
    df["release_fraction"].describe()
)


print("\nMain holding duration:")
print(
    df["main_holding_steps"].describe()
)


print("\nDominance ratio:")
finite_dominance = df[
    np.isfinite(df["dominance_ratio"])
]["dominance_ratio"]

print(
    finite_dominance.describe()
)


# Suspicious cases
suspicious = df[
    (df["main_holding_steps"] < 10)
    |
    (df["grasp_fraction"] < 0.15)
    |
    (df["grasp_fraction"] > 0.65)
    |
    (
        df["release_fraction"].notna()
        &
        (df["release_fraction"] < 0.65)
    )
]


print("\nPotentially suspicious after robust interval selection:")
print(len(suspicious))


if len(suspicious) > 0:

    print(
        suspicious[
            [
                "task_file",
                "demo",
                "num_steps",
                "num_closed_intervals",
                "main_grasp",
                "main_release",
                "main_holding_steps",
                "grasp_fraction",
                "release_fraction",
                "dominance_ratio",
            ]
        ].head(30)
    )