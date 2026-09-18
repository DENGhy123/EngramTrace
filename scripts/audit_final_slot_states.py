from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = (
    PROJECT_ROOT
    / "data"
    / "libero"
    / "libero_spatial"
)

PAIR_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "final_slots"
    / "final_slot_pairs.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "slot_state_audit"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# Settings
# ============================================================

N_RANDOM_PERMUTATIONS = 1000

RANDOM_SEED = 42


# ============================================================
# Helpers
# ============================================================

def task_to_file(task):

    file_path = (
        DATA_ROOT
        / f"{task}_demo.hdf5"
    )

    if not file_path.exists():

        raise FileNotFoundError(
            f"Cannot find:\n{file_path}"
        )

    return file_path


def extract_state(
    h5_file,
    demo_name,
    branch_step,
):
    """
    Extract the simulator state immediately BEFORE
    the behavioral branch.

    branch_step is the first step belonging to
    the successor primitive, so branch_step - 1
    is used conservatively here.
    """

    demo = (
        h5_file["data"][demo_name]
    )

    states = np.asarray(
        demo["states"],
        dtype=np.float64,
    )

    t = max(
        int(branch_step) - 1,
        0
    )

    t = min(
        t,
        len(states) - 1
    )

    return states[t]


# ============================================================
# Load
# ============================================================

pairs = pd.read_csv(
    PAIR_FILE
)

print("=" * 80)

print(
    "Final matched pairs:",
    len(pairs)
)

print(
    "Slots:",
    pairs["slot_id"].nunique()
)

print("=" * 80)


# ============================================================
# Extract simulator states
# ============================================================

records = []


for task, task_pairs in pairs.groupby(
    "task"
):

    file_path = task_to_file(
        task
    )

    print(
        "\nProcessing:",
        task
    )

    with h5py.File(
        file_path,
        "r"
    ) as f:

        for _, row in task_pairs.iterrows():

            state_a = extract_state(
                f,
                row["version_0_demo"],
                row["a_branch_step"],
            )

            state_b = extract_state(
                f,
                row["version_1_demo"],
                row["b_branch_step"],
            )

            records.append(
                {
                    "slot_id":
                        row["slot_id"],

                    "pair_id":
                        row["pair_id"],

                    "task":
                        task,

                    "version_0_demo":
                        row["version_0_demo"],

                    "version_1_demo":
                        row["version_1_demo"],

                    "a_branch_step":
                        row["a_branch_step"],

                    "b_branch_step":
                        row["b_branch_step"],

                    "state_a":
                        state_a,

                    "state_b":
                        state_b,
                }
            )


# ============================================================
# Standardize simulator state within each task
# ============================================================

pair_rows = []

summary_rows = []

rng = np.random.default_rng(
    RANDOM_SEED
)


for task in sorted(
    set(
        x["task"]
        for x in records
    )
):

    task_records = [
        x
        for x in records
        if x["task"] == task
    ]


    all_states = []

    for r in task_records:

        all_states.append(
            r["state_a"]
        )

        all_states.append(
            r["state_b"]
        )


    all_states = np.stack(
        all_states
    )


    scaler = StandardScaler()

    scaler.fit(
        all_states
    )


    # --------------------------------------------------------
    # Process each slot separately
    # --------------------------------------------------------

    slot_ids = sorted(
        set(
            r["slot_id"]
            for r in task_records
        )
    )


    for slot_id in slot_ids:

        slot_records = [
            r
            for r in task_records
            if r["slot_id"] == slot_id
        ]


        A = np.stack(
            [
                scaler.transform(
                    r["state_a"][None, :]
                )[0]
                for r in slot_records
            ]
        )


        B = np.stack(
            [
                scaler.transform(
                    r["state_b"][None, :]
                )[0]
                for r in slot_records
            ]
        )


        # ====================================================
        # Real matched distances
        # ====================================================

        matched_distances = np.linalg.norm(
            A - B,
            axis=1
        )


        for r, d in zip(
            slot_records,
            matched_distances
        ):

            raw_distance = np.linalg.norm(
                r["state_a"]
                -
                r["state_b"]
            )


            pair_rows.append(
                {
                    "slot_id":
                        slot_id,

                    "pair_id":
                        r["pair_id"],

                    "task":
                        task,

                    "version_0_demo":
                        r["version_0_demo"],

                    "version_1_demo":
                        r["version_1_demo"],

                    "full_state_raw_distance":
                        float(
                            raw_distance
                        ),

                    "full_state_z_distance":
                        float(
                            d
                        ),
                }
            )


        # ====================================================
        # Random pairing baseline
        # ====================================================

        random_means = []


        for _ in range(
            N_RANDOM_PERMUTATIONS
        ):

            perm = rng.permutation(
                len(B)
            )

            random_distances = np.linalg.norm(
                A
                -
                B[perm],
                axis=1
            )


            random_means.append(
                np.mean(
                    random_distances
                )
            )


        matched_mean = float(
            np.mean(
                matched_distances
            )
        )

        matched_median = float(
            np.median(
                matched_distances
            )
        )

        random_mean = float(
            np.mean(
                random_means
            )
        )


        state_matching_gain = (

            1.0

            -

            matched_mean
            /
            random_mean

            if random_mean > 0

            else np.nan
        )


        percentile = float(
            np.mean(
                np.array(
                    random_means
                )
                <=
                matched_mean
            )
        )


        summary_rows.append(
            {
                "slot_id":
                    slot_id,

                "task":
                    task,

                "num_pairs":
                    len(slot_records),

                "mean_full_state_distance":
                    matched_mean,

                "median_full_state_distance":
                    matched_median,

                "random_full_state_mean":
                    random_mean,

                "full_state_matching_gain":
                    state_matching_gain,

                "random_percentile":
                    percentile,
            }
        )


# ============================================================
# Save
# ============================================================

pair_df = pd.DataFrame(
    pair_rows
)

summary_df = pd.DataFrame(
    summary_rows
)


summary_df = summary_df.sort_values(
    "full_state_matching_gain",
    ascending=False
)


pair_path = (
    OUTPUT_DIR
    / "slot_full_state_pair_audit.csv"
)

summary_path = (
    OUTPUT_DIR
    / "slot_full_state_summary.csv"
)


pair_df.to_csv(
    pair_path,
    index=False
)

summary_df.to_csv(
    summary_path,
    index=False
)


# ============================================================
# Print
# ============================================================

print(
    "\n"
    +
    "=" * 100
)

print(
    "Full Simulator-State Slot Audit"
)

print(
    "=" * 100
)


print(
    summary_df
    .round(4)
    .to_string(
        index=False
    )
)


print(
    "\nSaved:"
)

print(
    pair_path
)

print(
    summary_path
)