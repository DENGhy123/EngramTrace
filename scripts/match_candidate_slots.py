from pathlib import Path
import json

import h5py
import numpy as np
import pandas as pd

from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import StandardScaler


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = (
    PROJECT_ROOT
    / "data"
    / "libero"
    / "libero_spatial"
)

CANDIDATE_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "candidate_slots"
)

CANDIDATE_FILE = (
    CANDIDATE_DIR
    / "candidate_slots.csv"
)

CHUNK_FILE = (
    CANDIDATE_DIR
    / "contextualized_motion_chunks.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "matched_slots"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# Number of random pairings used only as
# a comparison baseline.
N_RANDOM_PAIRINGS = 500

RANDOM_SEED = 42


# ============================================================
# Helpers
# ============================================================

def parse_demo_list(value):

    if pd.isna(value):
        return []

    value = str(value).strip()

    if not value:
        return []

    return [
        x
        for x in value.split(";")
        if x
    ]


def build_compressed_runs(group):
    """
    Convert chunk sequence into run-length-compressed
    primitive runs while retaining temporal boundaries.

    Example:

        P7 P7 P7 P0 P0 P1

    becomes

        [
            P7: start-end,
            P0: start-end,
            P1: start-end
        ]
    """

    group = group.sort_values(
        [
            "start",
            "chunk_index",
        ]
    )

    runs = []

    for _, row in group.iterrows():

        primitive = int(
            row["primitive"]
        )

        start = int(
            row["start"]
        )

        end = int(
            row["end"]
        )

        if (
            len(runs) == 0
            or
            runs[-1]["primitive"]
            !=
            primitive
        ):

            runs.append(
                {
                    "primitive":
                        primitive,

                    "start":
                        start,

                    "end":
                        end,
                }
            )

        else:

            runs[-1]["end"] = end

    return runs


def locate_transition(
    group,
    source,
    successor,
):
    """
    Locate source -> successor transition.

    Returns:
        branch_step
        occurrence_count

    branch_step is the first timestep of
    the successor primitive.
    """

    runs = build_compressed_runs(
        group
    )

    matches = []

    for i in range(
        len(runs) - 1
    ):

        if (
            runs[i]["primitive"]
            ==
            source

            and

            runs[i + 1]["primitive"]
            ==
            successor
        ):

            matches.append(
                {
                    "source_start":
                        runs[i]["start"],

                    "source_end":
                        runs[i]["end"],

                    "successor_start":
                        runs[i + 1]["start"],

                    "successor_end":
                        runs[i + 1]["end"],
                }
            )

    if len(matches) == 0:

        return None, 0

    # Current dataset almost always has one occurrence.
    # If several exist, use the earliest one and record
    # occurrence_count for later audit.
    chosen = matches[0]

    branch_step = (
        chosen[
            "successor_start"
        ]
    )

    return (
        branch_step,
        len(matches)
    )


def resolve_task_file(
    task,
    chunks_df,
):

    matches = (
        chunks_df[
            chunks_df["task"]
            ==
            task
        ]["task_file"]
        .dropna()
        .unique()
    )

    if len(matches) != 1:

        raise RuntimeError(
            f"Expected one HDF5 file for task "
            f"{task}, got {matches}"
        )

    file_path = (
        DATA_ROOT
        /
        matches[0]
    )

    if not file_path.exists():

        raise FileNotFoundError(
            f"Cannot find:\n{file_path}"
        )

    return file_path


# ============================================================
# Extract branch-point state
# ============================================================

def extract_demo_branch_features(
    task,
    demo_name,
    context,
    source,
    successor,
    chunks_df,
    h5_file,
):
    """
    Extract state BEFORE the two A/B versions diverge.

    We deliberately avoid using successor-motion features,
    because that would make matching depend on the
    fingerprint difference itself.
    """

    group = chunks_df[
        (chunks_df["task"] == task)
        &
        (chunks_df["demo"] == demo_name)
        &
        (chunks_df["context"] == context)
    ].copy()


    if len(group) == 0:

        return None


    branch_step, occurrence_count = (
        locate_transition(
            group,
            source,
            successor,
        )
    )


    if branch_step is None:

        return None


    demo = (
        h5_file[
            "data"
        ][
            demo_name
        ]
    )


    actions = np.asarray(
        demo["actions"]
    )

    obs = demo["obs"]

    T = len(
        actions
    )


    # Clamp to valid observation index.
    t = min(
        int(branch_step),
        T - 1,
    )


    ee_pos = np.asarray(
        obs["ee_pos"][t],
        dtype=np.float64,
    )

    ee_ori = np.asarray(
        obs["ee_ori"][t],
        dtype=np.float64,
    )

    joint_states = np.asarray(
        obs["joint_states"][t],
        dtype=np.float64,
    )

    gripper_states = np.asarray(
        obs["gripper_states"][t],
        dtype=np.float64,
    )


    # --------------------------------------------------------
    # Context timing
    # --------------------------------------------------------

    context_start = int(
        group["start"].min()
    )

    context_end = int(
        group["end"].max()
    )

    context_duration = max(
        context_end
        -
        context_start,
        1,
    )


    context_progress = (
        t
        -
        context_start
    ) / context_duration


    branch_fraction = (
        t / T
    )


    # --------------------------------------------------------
    # Metadata only
    # --------------------------------------------------------

    reward_last = (
        float(
            demo["rewards"][-1]
        )
        if "rewards" in demo
        else np.nan
    )

    reward_max = (
        float(
            np.max(
                demo["rewards"][:]
            )
        )
        if "rewards" in demo
        else np.nan
    )

    done_last = (
        float(
            demo["dones"][-1]
        )
        if "dones" in demo
        else np.nan
    )


    result = {

        "task":
            task,

        "demo":
            demo_name,

        "context":
            context,

        "source":
            source,

        "successor":
            successor,

        "branch_step":
            t,

        "branch_fraction":
            branch_fraction,

        "context_start":
            context_start,

        "context_end":
            context_end,

        "context_duration":
            context_duration,

        "context_progress":
            context_progress,

        "trajectory_length":
            T,

        "transition_occurrences":
            occurrence_count,

        "reward_last":
            reward_last,

        "reward_max":
            reward_max,

        "done_last":
            done_last,
    }


    # --------------------------------------------------------
    # Observable state features
    # --------------------------------------------------------

    for i in range(3):

        result[
            f"ee_pos_{i}"
        ] = ee_pos[i]


    for i in range(3):

        result[
            f"ee_ori_{i}"
        ] = ee_ori[i]


    for i in range(
        len(joint_states)
    ):

        result[
            f"joint_{i}"
        ] = joint_states[i]


    for i in range(
        len(gripper_states)
    ):

        result[
            f"gripper_{i}"
        ] = gripper_states[i]


    return result


# ============================================================
# Pairing
# ============================================================

def match_versions(
    a_df,
    b_df,
):
    """
    Hungarian one-to-one matching using pre-branch state.
    """

    feature_columns = [

        "ee_pos_0",
        "ee_pos_1",
        "ee_pos_2",

        "ee_ori_0",
        "ee_ori_1",
        "ee_ori_2",

        "joint_0",
        "joint_1",
        "joint_2",
        "joint_3",
        "joint_4",
        "joint_5",
        "joint_6",

        "gripper_0",
        "gripper_1",

        "branch_fraction",

        "context_progress",
    ]


    combined = pd.concat(
        [
            a_df,
            b_df,
        ],
        ignore_index=True,
    )


    scaler = StandardScaler()

    scaler.fit(
        combined[
            feature_columns
        ].to_numpy()
    )


    A = scaler.transform(
        a_df[
            feature_columns
        ].to_numpy()
    )

    B = scaler.transform(
        b_df[
            feature_columns
        ].to_numpy()
    )


    # --------------------------------------------------------
    # Pairwise Euclidean distance
    # --------------------------------------------------------

    cost_matrix = np.linalg.norm(
        A[:, None, :]
        -
        B[None, :, :],
        axis=2,
    )


    row_idx, col_idx = (
        linear_sum_assignment(
            cost_matrix
        )
    )


    return (
        row_idx,
        col_idx,
        cost_matrix,
        feature_columns,
    )


# ============================================================
# Load files
# ============================================================

candidate_df = pd.read_csv(
    CANDIDATE_FILE
)

chunks_df = pd.read_csv(
    CHUNK_FILE
)


print("=" * 80)

print(
    "Candidate families:",
    len(candidate_df)
)

print("=" * 80)


# ============================================================
# Main loop
# ============================================================

pair_rows = []

summary_rows = []


rng = np.random.default_rng(
    RANDOM_SEED
)


for candidate_index, candidate in (
    candidate_df.iterrows()
):

    candidate_id = (
        candidate[
            "candidate_id"
        ]
    )

    task = (
        candidate[
            "task"
        ]
    )

    context = (
        candidate[
            "context"
        ]
    )

    source = int(
        candidate[
            "source_primitive"
        ]
    )

    successor_a = int(
        candidate[
            "version_a_successor"
        ]
    )

    successor_b = int(
        candidate[
            "version_b_successor"
        ]
    )


    demos_a = parse_demo_list(
        candidate[
            "version_a_demos"
        ]
    )

    demos_b = parse_demo_list(
        candidate[
            "version_b_demos"
        ]
    )


    print(
        f"\n[{candidate_index + 1}/"
        f"{len(candidate_df)}] "
        f"{candidate_id}"
    )

    print(
        f"  {context}: "
        f"P{source}->P{successor_a} "
        f"vs "
        f"P{source}->P{successor_b}"
    )


    h5_path = resolve_task_file(
        task,
        chunks_df,
    )


    records_a = []

    records_b = []


    with h5py.File(
        h5_path,
        "r"
    ) as h5_file:

        for demo_name in demos_a:

            result = (
                extract_demo_branch_features(
                    task,
                    demo_name,
                    context,
                    source,
                    successor_a,
                    chunks_df,
                    h5_file,
                )
            )

            if result is not None:

                result[
                    "version"
                ] = "A"

                records_a.append(
                    result
                )


        for demo_name in demos_b:

            result = (
                extract_demo_branch_features(
                    task,
                    demo_name,
                    context,
                    source,
                    successor_b,
                    chunks_df,
                    h5_file,
                )
            )

            if result is not None:

                result[
                    "version"
                ] = "B"

                records_b.append(
                    result
                )


    a_df = pd.DataFrame(
        records_a
    )

    b_df = pd.DataFrame(
        records_b
    )


    if (
        len(a_df) == 0
        or
        len(b_df) == 0
    ):

        print(
            "  Skipped: empty A or B."
        )

        continue


    (
        row_idx,
        col_idx,
        cost_matrix,
        feature_columns,
    ) = match_versions(
        a_df,
        b_df,
    )


    optimal_distances = []


    for pair_number, (
        i,
        j,
    ) in enumerate(
        zip(
            row_idx,
            col_idx
        ),
        start=1,
    ):

        a = a_df.iloc[i]

        b = b_df.iloc[j]

        match_distance = float(
            cost_matrix[
                i,
                j
            ]
        )


        optimal_distances.append(
            match_distance
        )


        ee_pos_distance = float(
            np.linalg.norm(
                np.array(
                    [
                        a["ee_pos_0"],
                        a["ee_pos_1"],
                        a["ee_pos_2"],
                    ]
                )
                -
                np.array(
                    [
                        b["ee_pos_0"],
                        b["ee_pos_1"],
                        b["ee_pos_2"],
                    ]
                )
            )
        )


        ee_ori_distance = float(
            np.linalg.norm(
                np.array(
                    [
                        a["ee_ori_0"],
                        a["ee_ori_1"],
                        a["ee_ori_2"],
                    ]
                )
                -
                np.array(
                    [
                        b["ee_ori_0"],
                        b["ee_ori_1"],
                        b["ee_ori_2"],
                    ]
                )
            )
        )


        joint_a = np.array(
            [
                a[
                    f"joint_{k}"
                ]
                for k in range(7)
            ]
        )

        joint_b = np.array(
            [
                b[
                    f"joint_{k}"
                ]
                for k in range(7)
            ]
        )


        joint_distance = float(
            np.linalg.norm(
                joint_a
                -
                joint_b
            )
        )


        pair_rows.append(
            {

                "candidate_id":
                    candidate_id,

                "pair_id":
                    (
                        f"{candidate_id}"
                        f"_P{pair_number:02d}"
                    ),

                "task":
                    task,

                "context":
                    context,

                "source_primitive":
                    source,

                "version_a_successor":
                    successor_a,

                "version_b_successor":
                    successor_b,

                "a_demo":
                    a["demo"],

                "b_demo":
                    b["demo"],

                "match_distance":
                    match_distance,

                "ee_pos_distance":
                    ee_pos_distance,

                "ee_ori_distance":
                    ee_ori_distance,

                "joint_distance":
                    joint_distance,

                "branch_fraction_diff":
                    abs(
                        a[
                            "branch_fraction"
                        ]
                        -
                        b[
                            "branch_fraction"
                        ]
                    ),

                "context_progress_diff":
                    abs(
                        a[
                            "context_progress"
                        ]
                        -
                        b[
                            "context_progress"
                        ]
                    ),

                "trajectory_length_diff":
                    abs(
                        a[
                            "trajectory_length"
                        ]
                        -
                        b[
                            "trajectory_length"
                        ]
                    ),

                "context_duration_diff":
                    abs(
                        a[
                            "context_duration"
                        ]
                        -
                        b[
                            "context_duration"
                        ]
                    ),

                "a_branch_step":
                    int(
                        a[
                            "branch_step"
                        ]
                    ),

                "b_branch_step":
                    int(
                        b[
                            "branch_step"
                        ]
                    ),

                "a_transition_occurrences":
                    int(
                        a[
                            "transition_occurrences"
                        ]
                    ),

                "b_transition_occurrences":
                    int(
                        b[
                            "transition_occurrences"
                        ]
                    ),

                "a_reward_max":
                    a[
                        "reward_max"
                    ],

                "b_reward_max":
                    b[
                        "reward_max"
                    ],

                "a_done_last":
                    a[
                        "done_last"
                    ],

                "b_done_last":
                    b[
                        "done_last"
                    ],
            }
        )


    # ========================================================
    # Random-pairing baseline
    # ========================================================

    n_pairs = len(
        row_idx
    )


    random_means = []


    for _ in range(
        N_RANDOM_PAIRINGS
    ):

        if len(a_df) <= len(b_df):

            a_indices = np.arange(
                len(a_df)
            )

            b_indices = rng.choice(
                len(b_df),
                size=len(a_indices),
                replace=False,
            )

        else:

            b_indices = np.arange(
                len(b_df)
            )

            a_indices = rng.choice(
                len(a_df),
                size=len(b_indices),
                replace=False,
            )


        rng.shuffle(
            b_indices
        )


        random_cost = np.mean(
            cost_matrix[
                a_indices,
                b_indices,
            ]
        )


        random_means.append(
            random_cost
        )


    optimal_mean = float(
        np.mean(
            optimal_distances
        )
    )


    random_mean = float(
        np.mean(
            random_means
        )
    )


    matching_gain = (
        1.0
        -
        optimal_mean
        /
        random_mean
        if random_mean > 0
        else np.nan
    )


    summary_rows.append(
        {

            "candidate_id":
                candidate_id,

            "task":
                task,

            "context":
                context,

            "source_primitive":
                source,

            "version_a_successor":
                successor_a,

            "version_b_successor":
                successor_b,

            "num_a":
                len(a_df),

            "num_b":
                len(b_df),

            "num_pairs":
                n_pairs,

            "mean_match_distance":
                optimal_mean,

            "median_match_distance":
                float(
                    np.median(
                        optimal_distances
                    )
                ),

            "max_match_distance":
                float(
                    np.max(
                        optimal_distances
                    )
                ),

            "random_pairing_mean":
                random_mean,

            "matching_gain":
                matching_gain,

            "multi_occurrence_a":
                int(
                    (
                        a_df[
                            "transition_occurrences"
                        ]
                        >
                        1
                    ).sum()
                ),

            "multi_occurrence_b":
                int(
                    (
                        b_df[
                            "transition_occurrences"
                        ]
                        >
                        1
                    ).sum()
                ),
        }
    )


# ============================================================
# Save
# ============================================================

pairs_df = pd.DataFrame(
    pair_rows
)

summary_df = pd.DataFrame(
    summary_rows
)


# Rank pairs inside each candidate:
# low distance = better match
pairs_df[
    "pair_rank"
] = (
    pairs_df
    .groupby(
        "candidate_id"
    )[
        "match_distance"
    ]
    .rank(
        method="first",
        ascending=True,
    )
    .astype(int)
)


pairs_df = pairs_df.sort_values(
    [
        "candidate_id",
        "pair_rank",
    ]
)


summary_df = summary_df.sort_values(
    [
        "matching_gain",
        "median_match_distance",
    ],
    ascending=[
        False,
        True,
    ],
)


pairs_path = (
    OUTPUT_DIR
    / "matched_trajectory_pairs.csv"
)

summary_path = (
    OUTPUT_DIR
    / "candidate_matching_summary.csv"
)


pairs_df.to_csv(
    pairs_path,
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
    "A/B Trajectory Matching Summary"
)

print(
    "=" * 100
)


display_columns = [

    "candidate_id",

    "context",

    "num_a",

    "num_b",

    "num_pairs",

    "mean_match_distance",

    "median_match_distance",

    "random_pairing_mean",

    "matching_gain",

    "multi_occurrence_a",

    "multi_occurrence_b",
]


print(
    summary_df[
        display_columns
    ]
    .round(4)
    .to_string(
        index=False
    )
)


print(
    "\nSaved:"
)

print(
    pairs_path
)

print(
    summary_path
)