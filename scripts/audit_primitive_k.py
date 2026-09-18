from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score,
)


# =========================================================
# Paths
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_CSV = (
    PROJECT_ROOT
    / "outputs"
    / "primitives"
    / "motion_chunks_with_primitives.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "primitive_audit"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# Settings
# =========================================================

K_VALUES = [
    6,
    8,
    10,
    12,
    16,
]

SEEDS = [
    0,
    1,
    2,
    3,
    4,
]

SILHOUETTE_SAMPLE = 4000


# =========================================================
# Feature definitions
# =========================================================

BASE_FEATURES = [

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
]


FULL_FEATURES = BASE_FEATURES + [

    "vertical_ratio",
    "movement_efficiency",
]


FEATURE_SETS = {

    "full":
        FULL_FEATURES,

    "no_ratio":
        BASE_FEATURES,
}


# =========================================================
# Load
# =========================================================

df = pd.read_csv(
    INPUT_CSV
)

print(
    "Number of chunks:",
    len(df)
)


# =========================================================
# Evaluation
# =========================================================

all_rows = []

stability_rows = []


rng = np.random.RandomState(
    42
)


if len(df) > SILHOUETTE_SAMPLE:

    silhouette_idx = rng.choice(
        len(df),
        size=SILHOUETTE_SAMPLE,
        replace=False
    )

else:

    silhouette_idx = np.arange(
        len(df)
    )


for feature_name, feature_columns in FEATURE_SETS.items():

    print("\n" + "=" * 80)

    print(
        "Feature set:",
        feature_name
    )

    print("=" * 80)

    X = (
        df[
            feature_columns
        ]
        .to_numpy()
    )

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(
        X
    )


    for K in K_VALUES:

        labels_by_seed = {}

        print(
            f"\nK = {K}"
        )

        for seed in SEEDS:

            model = KMeans(

                n_clusters=K,

                random_state=seed,

                n_init=20,
            )

            labels = model.fit_predict(
                X_scaled
            )

            labels_by_seed[
                seed
            ] = labels


            # ---------------------------------------------
            # Cluster sizes
            # ---------------------------------------------

            counts = np.bincount(
                labels,
                minlength=K
            )

            fractions = (
                counts
                /
                len(labels)
            )


            min_fraction = (
                fractions.min()
            )

            max_fraction = (
                fractions.max()
            )


            # ---------------------------------------------
            # Standard clustering metrics
            # ---------------------------------------------

            sample_labels = labels[
                silhouette_idx
            ]

            sample_X = X_scaled[
                silhouette_idx
            ]


            silhouette = (
                silhouette_score(
                    sample_X,
                    sample_labels
                )
            )


            db_score = (
                davies_bouldin_score(
                    X_scaled,
                    labels
                )
            )


            ch_score = (
                calinski_harabasz_score(
                    X_scaled,
                    labels
                )
            )


            all_rows.append(
                {

                    "feature_set":
                        feature_name,

                    "K":
                        K,

                    "seed":
                        seed,

                    "silhouette":
                        silhouette,

                    "davies_bouldin":
                        db_score,

                    "calinski_harabasz":
                        ch_score,

                    "min_cluster_fraction":
                        min_fraction,

                    "max_cluster_fraction":
                        max_fraction,

                    "smallest_cluster":
                        counts.min(),

                    "largest_cluster":
                        counts.max(),
                }
            )


        # =================================================
        # Cross-seed stability
        # =================================================

        ari_values = []

        for seed_a, seed_b in combinations(
            SEEDS,
            2
        ):

            ari = adjusted_rand_score(

                labels_by_seed[
                    seed_a
                ],

                labels_by_seed[
                    seed_b
                ],
            )

            ari_values.append(
                ari
            )


        stability_rows.append(
            {

                "feature_set":
                    feature_name,

                "K":
                    K,

                "mean_ARI":
                    np.mean(
                        ari_values
                    ),

                "std_ARI":
                    np.std(
                        ari_values
                    ),

                "min_ARI":
                    np.min(
                        ari_values
                    ),

                "max_ARI":
                    np.max(
                        ari_values
                    ),
            }
        )


# =========================================================
# Save detailed results
# =========================================================

results_df = pd.DataFrame(
    all_rows
)

stability_df = pd.DataFrame(
    stability_rows
)


results_path = (
    OUTPUT_DIR
    / "primitive_k_metrics.csv"
)

stability_path = (
    OUTPUT_DIR
    / "primitive_k_stability.csv"
)


results_df.to_csv(
    results_path,
    index=False
)

stability_df.to_csv(
    stability_path,
    index=False
)


# =========================================================
# Aggregate metrics
# =========================================================

summary = (

    results_df

    .groupby(
        [
            "feature_set",
            "K"
        ]
    )

    .agg(

        silhouette_mean=
            (
                "silhouette",
                "mean"
            ),

        silhouette_std=
            (
                "silhouette",
                "std"
            ),

        db_mean=
            (
                "davies_bouldin",
                "mean"
            ),

        ch_mean=
            (
                "calinski_harabasz",
                "mean"
            ),

        min_cluster_fraction_mean=
            (
                "min_cluster_fraction",
                "mean"
            ),

        max_cluster_fraction_mean=
            (
                "max_cluster_fraction",
                "mean"
            ),
    )

    .reset_index()
)


summary = summary.merge(

    stability_df,

    on=[
        "feature_set",
        "K"
    ]
)


summary_path = (
    OUTPUT_DIR
    / "primitive_k_summary.csv"
)


summary.to_csv(
    summary_path,
    index=False
)


print("\n" + "=" * 100)

print(
    "Primitive K Audit"
)

print("=" * 100)


print(
    summary.round(4).to_string(
        index=False
    )
)


# =========================================================
# Plot silhouette
# =========================================================

plt.figure(
    figsize=(8, 5)
)


for feature_name in FEATURE_SETS:

    subset = summary[
        summary[
            "feature_set"
        ]
        ==
        feature_name
    ]

    plt.plot(

        subset["K"],

        subset[
            "silhouette_mean"
        ],

        marker="o",

        label=feature_name,
    )


plt.xlabel(
    "Number of primitives K"
)

plt.ylabel(
    "Mean silhouette score"
)

plt.title(
    "Primitive Cluster Separation"
)

plt.legend()

plt.tight_layout()


plt.savefig(

    OUTPUT_DIR
    / "silhouette_vs_k.png",

    dpi=200
)

plt.close()


# =========================================================
# Plot ARI
# =========================================================

plt.figure(
    figsize=(8, 5)
)


for feature_name in FEATURE_SETS:

    subset = stability_df[
        stability_df[
            "feature_set"
        ]
        ==
        feature_name
    ]

    plt.plot(

        subset["K"],

        subset[
            "mean_ARI"
        ],

        marker="o",

        label=feature_name,
    )


plt.xlabel(
    "Number of primitives K"
)

plt.ylabel(
    "Mean pairwise ARI"
)

plt.title(
    "Primitive Stability Across Seeds"
)

plt.legend()

plt.tight_layout()


plt.savefig(

    OUTPUT_DIR
    / "stability_vs_k.png",

    dpi=200
)

plt.close()


print("\nSaved to:")

print(
    OUTPUT_DIR
)