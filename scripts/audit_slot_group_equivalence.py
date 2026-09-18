from pathlib import Path
from itertools import combinations

import h5py
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances


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
    / "slot_group_audit"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# Helpers
# ============================================================

def task_to_file(task):

    path = (
        DATA_ROOT
        / f"{task}_demo.hdf5"
    )

    if not path.exists():
        raise FileNotFoundError(path)

    return path


def get_state(
    h5_file,
    demo_name,
    t,
):

    states = np.asarray(
        h5_file["data"][demo_name]["states"],
        dtype=np.float64,
    )

    t = max(
        0,
        min(
            int(t),
            len(states) - 1,
        )
    )

    return states[t]


def fit_task_scaler(h5_file):
    """
    Use ALL states from this task.
    """

    all_states = []

    for demo_name in h5_file["data"].keys():

        x = np.asarray(
            h5_file["data"][demo_name]["states"],
            dtype=np.float64,
        )

        all_states.append(x)

    X = np.concatenate(
        all_states,
        axis=0,
    )

    scaler = StandardScaler()
    scaler.fit(X)

    return scaler


# ============================================================
# Statistics
# ============================================================

def centroid_distance(A, B):

    return float(
        np.linalg.norm(
            A.mean(axis=0)
            -
            B.mean(axis=0)
        )
    )


def compute_rbf_gamma(X):
    """
    Median heuristic.
    """

    D = pairwise_distances(
        X,
        metric="euclidean",
    )

    values = D[
        np.triu_indices_from(
            D,
            k=1,
        )
    ]

    positive = values[
        values > 0
    ]

    if len(positive) == 0:
        return 1.0

    median_distance = np.median(
        positive
    )

    gamma = (
        1.0
        /
        (
            2.0
            *
            median_distance ** 2
            +
            1e-12
        )
    )

    return gamma


def rbf_kernel(X, gamma):

    sq_dist = (
        pairwise_distances(
            X,
            metric="sqeuclidean",
        )
    )

    return np.exp(
        -gamma
        *
        sq_dist
    )


def mmd2_from_kernel(
    K,
    idx_a,
    idx_b,
):
    """
    Biased MMD^2.

    Non-negative and suitable here as
    a permutation statistic.
    """

    Kaa = K[
        np.ix_(
            idx_a,
            idx_a,
        )
    ]

    Kbb = K[
        np.ix_(
            idx_b,
            idx_b,
        )
    ]

    Kab = K[
        np.ix_(
            idx_a,
            idx_b,
        )
    ]

    return float(
        Kaa.mean()
        +
        Kbb.mean()
        -
        2.0 * Kab.mean()
    )


def exact_group_test(
    A,
    B,
):
    """
    Exact label-permutation test.

    5 vs 5:
        C(10,5) = 252 possible assignments.

    Larger statistic = stronger group separation.
    """

    X = np.concatenate(
        [A, B],
        axis=0,
    )

    n_a = len(A)
    n_total = len(X)

    observed_a = np.arange(
        n_a
    )

    observed_b = np.arange(
        n_a,
        n_total,
    )


    # --------------------------------------------------------
    # Observed centroid distance
    # --------------------------------------------------------

    obs_centroid = centroid_distance(
        A,
        B,
    )


    # --------------------------------------------------------
    # Observed MMD
    # --------------------------------------------------------

    gamma = compute_rbf_gamma(
        X
    )

    K = rbf_kernel(
        X,
        gamma,
    )


    obs_mmd = mmd2_from_kernel(
        K,
        observed_a,
        observed_b,
    )


    # --------------------------------------------------------
    # Exact label permutations
    # --------------------------------------------------------

    all_indices = np.arange(
        n_total
    )

    centroid_null = []
    mmd_null = []


    for combo in combinations(
        range(n_total),
        n_a,
    ):

        idx_a = np.asarray(
            combo
        )

        mask = np.ones(
            n_total,
            dtype=bool,
        )

        mask[idx_a] = False

        idx_b = all_indices[
            mask
        ]


        centroid_null.append(
            centroid_distance(
                X[idx_a],
                X[idx_b],
            )
        )


        mmd_null.append(
            mmd2_from_kernel(
                K,
                idx_a,
                idx_b,
            )
        )


    centroid_null = np.asarray(
        centroid_null
    )

    mmd_null = np.asarray(
        mmd_null
    )


    # Large statistic indicates more separation.
    centroid_p = (
        1
        +
        np.sum(
            centroid_null
            >=
            obs_centroid
        )
    ) / (
        len(centroid_null)
        +
        1
    )


    mmd_p = (
        1
        +
        np.sum(
            mmd_null
            >=
            obs_mmd
        )
    ) / (
        len(mmd_null)
        +
        1
    )


    # --------------------------------------------------------
    # Relative separation
    #
    # < 1 means observed A/B separation is
    # below a typical random relabeling.
    # --------------------------------------------------------

    centroid_ratio = (
        obs_centroid
        /
        centroid_null.mean()
    )


    mmd_ratio = (
        obs_mmd
        /
        mmd_null.mean()
    )


    # --------------------------------------------------------
    # Dimension-wise standardized mean differences
    # --------------------------------------------------------

    diff = np.abs(
        A.mean(axis=0)
        -
        B.mean(axis=0)
    )


    mean_abs_smd = float(
        diff.mean()
    )

    max_abs_smd = float(
        diff.max()
    )


    return {

        "centroid_distance":
            obs_centroid,

        "centroid_random_mean":
            float(
                centroid_null.mean()
            ),

        "centroid_ratio":
            centroid_ratio,

        "centroid_exact_p":
            centroid_p,

        "mmd2":
            obs_mmd,

        "mmd_random_mean":
            float(
                mmd_null.mean()
            ),

        "mmd_ratio":
            mmd_ratio,

        "mmd_exact_p":
            mmd_p,

        "mean_abs_smd":
            mean_abs_smd,

        "max_abs_smd":
            max_abs_smd,
    }


# ============================================================
# Load
# ============================================================

pairs = pd.read_csv(
    PAIR_FILE
)


results = []


# ============================================================
# Process each task / slot
# ============================================================

for task, task_pairs in pairs.groupby(
    "task"
):

    path = task_to_file(
        task
    )

    print(
        "\nProcessing task:",
        task
    )


    with h5py.File(
        path,
        "r"
    ) as f:

        scaler = fit_task_scaler(
            f
        )


        for slot_id, slot_pairs in (
            task_pairs.groupby(
                "slot_id"
            )
        ):

            slot_pairs = (
                slot_pairs
                .sort_values(
                    "slot_pair_index"
                )
            )


            initial_A = []
            initial_B = []

            branch_A = []
            branch_B = []


            for _, pair in slot_pairs.iterrows():

                # =================================================
                # Initial states
                # =================================================

                a_initial = get_state(
                    f,
                    pair[
                        "version_0_demo"
                    ],
                    0,
                )

                b_initial = get_state(
                    f,
                    pair[
                        "version_1_demo"
                    ],
                    0,
                )


                # =================================================
                # Pre-branch states
                # =================================================

                ta = max(
                    int(
                        pair[
                            "a_branch_step"
                        ]
                    )
                    -
                    1,
                    0,
                )

                tb = max(
                    int(
                        pair[
                            "b_branch_step"
                        ]
                    )
                    -
                    1,
                    0,
                )


                a_branch = get_state(
                    f,
                    pair[
                        "version_0_demo"
                    ],
                    ta,
                )

                b_branch = get_state(
                    f,
                    pair[
                        "version_1_demo"
                    ],
                    tb,
                )


                initial_A.append(
                    scaler.transform(
                        a_initial[
                            None,
                            :
                        ]
                    )[0]
                )

                initial_B.append(
                    scaler.transform(
                        b_initial[
                            None,
                            :
                        ]
                    )[0]
                )

                branch_A.append(
                    scaler.transform(
                        a_branch[
                            None,
                            :
                        ]
                    )[0]
                )

                branch_B.append(
                    scaler.transform(
                        b_branch[
                            None,
                            :
                        ]
                    )[0]
                )


            initial_A = np.stack(
                initial_A
            )

            initial_B = np.stack(
                initial_B
            )

            branch_A = np.stack(
                branch_A
            )

            branch_B = np.stack(
                branch_B
            )


            initial_result = exact_group_test(
                initial_A,
                initial_B,
            )

            branch_result = exact_group_test(
                branch_A,
                branch_B,
            )


            row = {

                "slot_id":
                    slot_id,

                "task":
                    task,

                "num_version_0":
                    len(initial_A),

                "num_version_1":
                    len(initial_B),
            }


            for key, value in initial_result.items():

                row[
                    f"initial_{key}"
                ] = value


            for key, value in branch_result.items():

                row[
                    f"branch_{key}"
                ] = value


            results.append(
                row
            )


# ============================================================
# Save
# ============================================================

result_df = pd.DataFrame(
    results
)


output_path = (
    OUTPUT_DIR
    / "slot_group_equivalence.csv"
)


result_df.to_csv(
    output_path,
    index=False
)


# ============================================================
# Compact display
# ============================================================

display_columns = [

    "slot_id",

    "initial_centroid_ratio",
    "initial_centroid_exact_p",

    "initial_mmd_ratio",
    "initial_mmd_exact_p",

    "branch_centroid_ratio",
    "branch_centroid_exact_p",

    "branch_mmd_ratio",
    "branch_mmd_exact_p",
]


print(
    "\n"
    +
    "=" * 120
)

print(
    "Slot Group-Level Equivalence Audit"
)

print(
    "=" * 120
)


print(
    result_df[
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
    output_path
)