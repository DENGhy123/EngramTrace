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


def find_gripper_switches(gripper):
    """
    Return transition positions.

    LIBERO / robosuite:
    -1 = open
    +1 = closed
    """

    diff = np.diff(gripper)

    close_events = np.where(diff > 0)[0] + 1
    open_events = np.where(diff < 0)[0] + 1

    return close_events, open_events


rows = []


files = sorted(DATA_ROOT.glob("*.hdf5"))

print(f"Found {len(files)} task files.")


for file_path in files:

    with h5py.File(file_path, "r") as f:

        instruction = f["data"].attrs.get(
            "problem_info",
            ""
        )

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

            close_events, open_events = (
                find_gripper_switches(gripper)
            )

            T = len(actions)

            # Real EE path length
            delta_pos = np.diff(
                ee_pos,
                axis=0
            )

            path_length = np.linalg.norm(
                delta_pos,
                axis=1
            ).sum()

            first_close = (
                int(close_events[0])
                if len(close_events) > 0
                else None
            )

            first_open_after_close = None

            if first_close is not None:

                valid_open = open_events[
                    open_events > first_close
                ]

                if len(valid_open) > 0:
                    first_open_after_close = int(
                        valid_open[0]
                    )

            grasp_fraction = (
                first_close / T
                if first_close is not None
                else np.nan
            )

            release_fraction = (
                first_open_after_close / T
                if first_open_after_close is not None
                else np.nan
            )

            holding_steps = (
                first_open_after_close
                - first_close
                if (
                    first_close is not None
                    and
                    first_open_after_close is not None
                )
                else np.nan
            )

            rows.append(
                {
                    "task_file": file_path.name,
                    "demo": demo_name,
                    "num_steps": T,
                    "num_close_events":
                        len(close_events),
                    "num_open_events":
                        len(open_events),
                    "first_close":
                        first_close,
                    "first_open_after_close":
                        first_open_after_close,
                    "grasp_fraction":
                        grasp_fraction,
                    "release_fraction":
                        release_fraction,
                    "holding_steps":
                        holding_steps,
                    "ee_path_length":
                        path_length,
                }
            )


df = pd.DataFrame(rows)


output_csv = (
    OUTPUT_DIR
    / "libero_spatial_demo_statistics.csv"
)

df.to_csv(
    output_csv,
    index=False
)


print("\nSaved:")
print(output_csv)


print("\n" + "=" * 70)
print("Dataset summary")
print("=" * 70)

print("Number of demonstrations:")
print(len(df))

print("\nTrajectory length:")
print(
    df["num_steps"].describe()
)

print("\nFirst close fraction:")
print(
    df["grasp_fraction"].describe()
)

print("\nRelease fraction:")
print(
    df["release_fraction"].describe()
)

print("\nHolding steps:")
print(
    df["holding_steps"].describe()
)

print("\nNumber of gripper close events:")
print(
    df["num_close_events"].value_counts()
)

print("\nNumber of gripper open events:")
print(
    df["num_open_events"].value_counts()
)


abnormal = df[
    (df["num_close_events"] != 1)
    |
    (df["num_open_events"] != 1)
]


print("\nPotentially unusual demonstrations:")
print(len(abnormal))

if len(abnormal) > 0:

    print(
        abnormal[
            [
                "task_file",
                "demo",
                "num_steps",
                "num_close_events",
                "num_open_events",
            ]
        ].head(20)
    )