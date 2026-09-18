from pathlib import Path
import json

import h5py
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# =========================================================
# Configuration
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = (
    PROJECT_ROOT
    / "data"
    / "libero"
    / "libero_spatial"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "primitives"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


CHUNK_SIZE = 5

# Do not keep extremely short remainder chunks
MIN_CHUNK_SIZE = 3

# First exploratory value.
# We will compare K later.
N_PRIMITIVES = 8

RANDOM_SEED = 42


# =========================================================
# Utility functions
# =========================================================

def demo_sort_key(name):
    return int(name.split("_")[1])


def find_closed_intervals(gripper):
    """
    Find all continuous intervals where gripper command > 0.

    Returns:
        [(start, end, length), ...]

    start/end are inclusive.
    """

    closed = (gripper > 0).astype(np.int32)

    intervals = []

    start = None

    for t, state in enumerate(closed):

        if state == 1 and start is None:
            start = t

        elif state == 0 and start is not None:

            end = t - 1

            intervals.append(
                (
                    start,
                    end,
                    end - start + 1
                )
            )

            start = None

    if start is not None:

        end = len(closed) - 1

        intervals.append(
            (
                start,
                end,
                end - start + 1
            )
        )

    return intervals


def get_main_grasp_interval(gripper):
    """
    Main grasp interval is defined as the
    longest continuously closed gripper interval.
    """

    intervals = find_closed_intervals(
        gripper
    )

    if not intervals:
        return None, None

    main = max(
        intervals,
        key=lambda x: x[2]
    )

    grasp = main[0]

    if main[1] < len(gripper) - 1:
        release = main[1] + 1
    else:
        release = None

    return grasp, release


def make_segments(T, grasp, release):
    """
    Create event-aware trajectory segments.

    Intervals follow Python convention:
        [start, end)
    """

    segments = []

    if grasp is None:

        segments.append(
            (
                "FULL_TRAJECTORY",
                0,
                T
            )
        )

        return segments

    # -------------------------
    # Pre-grasp
    # -------------------------

    if grasp > 0:

        segments.append(
            (
                "PRE_GRASP",
                0,
                grasp
            )
        )

    # -------------------------
    # Holding
    # -------------------------

    holding_end = (
        release
        if release is not None
        else T
    )

    if holding_end > grasp:

        segments.append(
            (
                "HOLDING",
                grasp,
                holding_end
            )
        )

    # -------------------------
    # Post-release
    # -------------------------

    if (
        release is not None
        and release < T
    ):

        segments.append(
            (
                "POST_RELEASE",
                release,
                T
            )
        )

    return segments


# =========================================================
# Feature extraction
# =========================================================

def extract_chunk_features(
    actions,
    ee_pos,
    start,
    end,
):
    """
    Extract interpretable motion features
    for one short trajectory chunk.

    Gripper is deliberately NOT used
    for motion clustering.
    """

    chunk_actions = actions[
        start:end,
        :6
    ]

    chunk_pos = ee_pos[
        start:end
    ]

    n = end - start

    # -----------------------------------------------------
    # 1. Mean normalized action commands
    # -----------------------------------------------------

    mean_xyz_command = np.mean(
        chunk_actions[:, :3],
        axis=0
    )

    mean_rot_command = np.mean(
        chunk_actions[:, 3:6],
        axis=0
    )

    # -----------------------------------------------------
    # 2. Command magnitude
    # -----------------------------------------------------

    translation_command_mag = np.mean(
        np.linalg.norm(
            chunk_actions[:, :3],
            axis=1
        )
    )

    rotation_command_mag = np.mean(
        np.linalg.norm(
            chunk_actions[:, 3:6],
            axis=1
        )
    )

    # -----------------------------------------------------
    # 3. Actual EE displacement
    # -----------------------------------------------------

    if len(chunk_pos) >= 2:

        displacement = (
            chunk_pos[-1]
            -
            chunk_pos[0]
        )

        step_displacements = np.diff(
            chunk_pos,
            axis=0
        )

        path_length = np.linalg.norm(
            step_displacements,
            axis=1
        ).sum()

        num_motion_steps = (
            len(chunk_pos) - 1
        )

    else:

        displacement = np.zeros(3)

        path_length = 0.0

        num_motion_steps = 1


    displacement_per_step = (
        displacement
        /
        max(num_motion_steps, 1)
    )

    path_per_step = (
        path_length
        /
        max(num_motion_steps, 1)
    )

    # -----------------------------------------------------
    # 4. Movement geometry
    # -----------------------------------------------------

    straight_distance = np.linalg.norm(
        displacement
    )

    horizontal_distance = np.linalg.norm(
        displacement[:2]
    )

    efficiency = (
        straight_distance
        /
        (path_length + 1e-8)
    )

    vertical_ratio = (
        abs(displacement[2])
        /
        (path_length + 1e-8)
    )

    # -----------------------------------------------------
    # Return physical features
    # -----------------------------------------------------

    features = {
        "cmd_dx": mean_xyz_command[0],
        "cmd_dy": mean_xyz_command[1],
        "cmd_dz": mean_xyz_command[2],

        "cmd_drx": mean_rot_command[0],
        "cmd_dry": mean_rot_command[1],
        "cmd_drz": mean_rot_command[2],

        "translation_command_mag":
            translation_command_mag,

        "rotation_command_mag":
            rotation_command_mag,

        "ee_dx_per_step":
            displacement_per_step[0],

        "ee_dy_per_step":
            displacement_per_step[1],

        "ee_dz_per_step":
            displacement_per_step[2],

        "ee_path_per_step":
            path_per_step,

        "horizontal_distance":
            horizontal_distance,

        "vertical_ratio":
            vertical_ratio,

        "movement_efficiency":
            efficiency,

        "chunk_length":
            n,
    }

    return features


# =========================================================
# Build chunk dataset
# =========================================================

rows = []

task_files = sorted(
    DATA_ROOT.glob("*.hdf5")
)

print(
    f"Found {len(task_files)} task files."
)


for task_index, file_path in enumerate(
    task_files
):

    print(
        f"[{task_index + 1}/{len(task_files)}] "
        f"{file_path.name}"
    )

    with h5py.File(
        file_path,
        "r"
    ) as f:

        problem_info = (
            f["data"]
            .attrs.get(
                "problem_info",
                ""
            )
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

            T = len(actions)

            grasp, release = (
                get_main_grasp_interval(
                    gripper
                )
            )

            segments = make_segments(
                T,
                grasp,
                release
            )

            chunk_global_index = 0

            for (
                phase,
                segment_start,
                segment_end
            ) in segments:

                start = segment_start

                while start < segment_end:

                    end = min(
                        start + CHUNK_SIZE,
                        segment_end
                    )

                    if (
                        end - start
                        <
                        MIN_CHUNK_SIZE
                    ):
                        break

                    features = (
                        extract_chunk_features(
                            actions,
                            ee_pos,
                            start,
                            end
                        )
                    )

                    # Gripper state is metadata,
                    # NOT a clustering feature
                    gripper_mean = np.mean(
                        actions[
                            start:end,
                            6
                        ]
                    )

                    gripper_state = (
                        "CLOSED"
                        if gripper_mean > 0
                        else "OPEN"
                    )

                    row = {
                        "task_file":
                            file_path.name,

                        "demo":
                            demo_name,

                        "phase":
                            phase,

                        "chunk_index":
                            chunk_global_index,

                        "start":
                            start,

                        "end":
                            end,

                        "trajectory_length":
                            T,

                        "grasp_step":
                            grasp,

                        "release_step":
                            release,

                        "gripper_state":
                            gripper_state,
                    }

                    row.update(
                        features
                    )

                    rows.append(
                        row
                    )

                    chunk_global_index += 1

                    start = end


df = pd.DataFrame(rows)


print("\nTotal motion chunks:")
print(len(df))


# =========================================================
# Features used for clustering
# =========================================================

FEATURE_COLUMNS = [

    "cmd_dx",
    "cmd_dy",
    "cmd_dz",

    "cmd_drx",
    "cmd_dry",
    "cmd_drz",

    "translation_command_mag",
    "rotation_command_mag",

    "ee_dx_per_step",
    "ee_dy_per_step",
    "ee_dz_per_step",

    "ee_path_per_step",

    "vertical_ratio",

    "movement_efficiency",
]


X = df[
    FEATURE_COLUMNS
].to_numpy()


# =========================================================
# Normalize
# =========================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(
    X
)


# =========================================================
# K-means
# =========================================================

kmeans = KMeans(
    n_clusters=N_PRIMITIVES,
    random_state=RANDOM_SEED,
    n_init=20,
)

primitive_labels = (
    kmeans.fit_predict(
        X_scaled
    )
)

df["primitive"] = (
    primitive_labels
)


# =========================================================
# Save models
# =========================================================

joblib.dump(
    scaler,
    OUTPUT_DIR
    / "primitive_scaler.joblib"
)

joblib.dump(
    kmeans,
    OUTPUT_DIR
    / "primitive_kmeans.joblib"
)


# =========================================================
# Save chunk table
# =========================================================

chunk_csv = (
    OUTPUT_DIR
    / "motion_chunks_with_primitives.csv"
)

df.to_csv(
    chunk_csv,
    index=False
)


print("\nSaved:")
print(chunk_csv)


# =========================================================
# Cluster summary
# =========================================================

cluster_summary = (
    df
    .groupby("primitive")[
        FEATURE_COLUMNS
    ]
    .mean()
)


cluster_counts = (
    df["primitive"]
    .value_counts()
    .sort_index()
)


cluster_summary.insert(
    0,
    "count",
    cluster_counts
)


cluster_summary_path = (
    OUTPUT_DIR
    / "primitive_cluster_summary.csv"
)

cluster_summary.to_csv(
    cluster_summary_path
)


print("\nSaved:")
print(cluster_summary_path)


print("\n" + "=" * 80)
print("Primitive cluster sizes")
print("=" * 80)

print(
    cluster_counts
)


print("\n" + "=" * 80)
print("Primitive cluster mean motion")
print("=" * 80)

display_columns = [

    "cmd_dx",
    "cmd_dy",
    "cmd_dz",

    "ee_dx_per_step",
    "ee_dy_per_step",
    "ee_dz_per_step",

    "translation_command_mag",
    "rotation_command_mag",

    "vertical_ratio",
]

print(
    cluster_summary[
        display_columns
    ].round(4)
)


# =========================================================
# Phase distribution per primitive
# =========================================================

phase_distribution = pd.crosstab(
    df["primitive"],
    df["phase"],
    normalize="index"
)


phase_path = (
    OUTPUT_DIR
    / "primitive_phase_distribution.csv"
)

phase_distribution.to_csv(
    phase_path
)


print("\n" + "=" * 80)
print("Phase distribution of primitives")
print("=" * 80)

print(
    phase_distribution.round(3)
)


# =========================================================
# Build primitive sequences
# =========================================================

sequence_dict = {}


grouped = df.groupby(
    [
        "task_file",
        "demo"
    ],
    sort=False
)


for (
    task_file,
    demo_name
), group in grouped:

    group = group.sort_values(
        "chunk_index"
    )

    raw_sequence = (
        group["primitive"]
        .astype(int)
        .tolist()
    )

    phases = (
        group["phase"]
        .tolist()
    )

    # Run-length compressed primitive sequence
    compressed = []

    previous = None

    for primitive in raw_sequence:

        if primitive != previous:

            compressed.append(
                primitive
            )

            previous = primitive

    key = (
        f"{task_file}::{demo_name}"
    )

    sequence_dict[key] = {
        "raw_sequence":
            raw_sequence,

        "compressed_sequence":
            compressed,

        "phases":
            phases,
    }


sequence_path = (
    OUTPUT_DIR
    / "primitive_sequences.json"
)


with open(
    sequence_path,
    "w"
) as f:

    json.dump(
        sequence_dict,
        f,
        indent=2
    )


print("\nSaved:")
print(sequence_path)


# =========================================================
# PCA visualization
# =========================================================

pca = PCA(
    n_components=2,
    random_state=RANDOM_SEED
)

X_pca = pca.fit_transform(
    X_scaled
)


plt.figure(
    figsize=(10, 8)
)


for primitive in range(
    N_PRIMITIVES
):

    mask = (
        primitive_labels
        ==
        primitive
    )

    plt.scatter(
        X_pca[mask, 0],
        X_pca[mask, 1],
        s=8,
        alpha=0.4,
        label=f"P{primitive}"
    )


plt.xlabel(
    "PCA component 1"
)

plt.ylabel(
    "PCA component 2"
)

plt.title(
    "LIBERO Motion Primitive Clusters"
)

plt.legend(
    ncol=2
)

plt.tight_layout()


pca_path = (
    OUTPUT_DIR
    / "primitive_clusters_pca.png"
)

plt.savefig(
    pca_path,
    dpi=200
)

plt.close()


print("\nSaved:")
print(pca_path)


print("\nDone.")