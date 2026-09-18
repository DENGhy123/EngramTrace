from pathlib import Path
from itertools import permutations

import h5py
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler


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
    / "slot_state_audit_v2"
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


def fit_task_scaler(h5_file):
    """
    Fit state normalization using ALL simulator states
    from all demonstrations in this task.
    """

    all_states = []

    for demo_name in h5_file["data"].keys():

        states = np.asarray(
            h5_file["data"][demo_name]["states"],
            dtype=np.float64,
        )

        all_states.append(states)


    X = np.concatenate(
        all_states,
        axis=0,
    )


    scaler = StandardScaler()

    scaler.fit(X)

    return scaler


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


def exact_permutation_test(
    A,
    B,
):
    """
    Compute exact permutation distribution.

    n = 5 -> 5! = 120 permutations.
    """

    n = len(B)

    observed_distances = np.linalg.norm(
        A - B,
        axis=1,
    )

    observed_mean = float(
        observed_distances.mean()
    )


    perm_means = []

    for perm in permutations(
        range(n)
    ):

        perm = np.asarray(perm)

        distances = np.linalg.norm(
            A - B[perm],
            axis=1,
        )

        perm_means.append(
            distances.mean()
        )


    perm_means = np.asarray(
        perm_means
    )


    # Smaller distance = better matching.
    #
    # +1 correction gives a conservative finite
    # permutation p-value.
    p_value = (

        1
        +
        np.sum(
            perm_means
            <=
            observed_mean
        )

    ) / (

        len(perm_means)
        +
        1
    )


    random_mean = float(
        perm_means.mean()
    )


    gain = (

        1.0
        -
        observed_mean
        /
        random_mean

        if random_mean > 0

        else np.nan
    )


    return {

        "observed_mean":
            observed_mean,

        "observed_median":
            float(
                np.median(
                    observed_distances
                )
            ),

        "permutation_mean":
            random_mean,

        "matching_gain":
            gain,

        "exact_p_value":
            p_value,

        "min_distance":
            float(
                observed_distances.min()
            ),

        "max_distance":
            float(
                observed_distances.max()
            ),
    }


# ============================================================
# Load
# ============================================================

pairs = pd.read_csv(
    PAIR_FILE
)


rows = []


# ============================================================
# Process per task
# ============================================================

for task, task_pairs in pairs.groupby(
    "task"
):

    print(
        "\nProcessing:",
        task
    )

    path = task_to_file(
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


            init_A = []
            init_B = []

            branch_A = []
            branch_B = []


            for _, pair in (
                slot_pairs.iterrows()
            ):

                # --------------------------------------------
                # Initial state
                # --------------------------------------------

                a0 = get_state(
                    f,
                    pair["version_0_demo"],
                    0,
                )

                b0 = get_state(
                    f,
                    pair["version_1_demo"],
                    0,
                )


                # --------------------------------------------
                # Pre-branch state
                # --------------------------------------------

                ta = max(
                    int(
                        pair["a_branch_step"]
                    )
                    -
                    1,
                    0,
                )

                tb = max(
                    int(
                        pair["b_branch_step"]
                    )
                    -
                    1,
                    0,
                )


                ab = get_state(
                    f,
                    pair["version_0_demo"],
                    ta,
                )

                bb = get_state(
                    f,
                    pair["version_1_demo"],
                    tb,
                )


                init_A.append(
                    scaler.transform(
                        a0[None, :]
                    )[0]
                )

                init_B.append(
                    scaler.transform(
                        b0[None, :]
                    )[0]
                )

                branch_A.append(
                    scaler.transform(
                        ab[None, :]
                    )[0]
                )

                branch_B.append(
                    scaler.transform(
                        bb[None, :]
                    )[0]
                )


            init_A = np.stack(
                init_A
            )

            init_B = np.stack(
                init_B
            )

            branch_A = np.stack(
                branch_A
            )

            branch_B = np.stack(
                branch_B
            )


            initial_result = (
                exact_permutation_test(
                    init_A,
                    init_B,
                )
            )


            branch_result = (
                exact_permutation_test(
                    branch_A,
                    branch_B,
                )
            )


            rows.append(
                {

                    "slot_id":
                        slot_id,

                    "task":
                        task,

                    "num_pairs":
                        len(slot_pairs),

                    # Initial scene/state
                    "initial_mean_distance":
                        initial_result[
                            "observed_mean"
                        ],

                    "initial_random_mean":
                        initial_result[
                            "permutation_mean"
                        ],

                    "initial_matching_gain":
                        initial_result[
                            "matching_gain"
                        ],

                    "initial_exact_p":
                        initial_result[
                            "exact_p_value"
                        ],

                    # Pre-branch state
                    "branch_mean_distance":
                        branch_result[
                            "observed_mean"
                        ],

                    "branch_random_mean":
                        branch_result[
                            "permutation_mean"
                        ],

                    "branch_matching_gain":
                        branch_result[
                            "matching_gain"
                        ],

                    "branch_exact_p":
                        branch_result[
                            "exact_p_value"
                        ],
                }
            )


# ============================================================
# Save
# ============================================================

result_df = pd.DataFrame(
    rows
)


result_df = result_df.sort_values(
    [
        "initial_matching_gain",
        "branch_matching_gain",
    ],
    ascending=False,
)


output_path = (
    OUTPUT_DIR
    / "slot_state_equivalence_v2.csv"
)


result_df.to_csv(
    output_path,
    index=False,
)


print(
    "\n"
    +
    "=" * 110
)

print(
    "Slot State Equivalence V2"
)

print(
    "=" * 110
)


print(
    result_df
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