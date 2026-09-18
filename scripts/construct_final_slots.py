from pathlib import Path
import json

import numpy as np
import pandas as pd


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CANDIDATE_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "candidate_slots"
)

MATCH_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "matched_slots"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "final_slots"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


CANDIDATE_FILE = (
    CANDIDATE_DIR
    / "candidate_slots.csv"
)

MATCH_SUMMARY_FILE = (
    MATCH_DIR
    / "candidate_matching_summary.csv"
)

PAIR_FILE = (
    MATCH_DIR
    / "matched_trajectory_pairs.csv"
)


# ============================================================
# Configuration
# ============================================================

# Every final slot has exactly:
#
# 5 trajectories in Version 0
# 5 matched trajectories in Version 1
#
BUNDLE_SIZE = 5


# ------------------------------------------------------------
# Pair-level quality thresholds
# ------------------------------------------------------------

MAX_MATCH_DISTANCE = 4.30

MAX_EE_POS_DISTANCE = 0.05

MAX_EE_ORI_DISTANCE = 0.30

MAX_JOINT_DISTANCE = 0.35

MAX_BRANCH_FRACTION_DIFF = 0.06

MAX_CONTEXT_PROGRESS_DIFF = 0.20

MAX_TRAJECTORY_LENGTH_DIFF = 20


# ------------------------------------------------------------
# Candidate-family thresholds
# ------------------------------------------------------------

MIN_GOOD_PAIRS = BUNDLE_SIZE

MIN_MATCHING_GAIN = 0.35

MIN_BALANCE_SCORE = 0.50

MIN_CLEAN_COVERAGE = 0.70

MAX_OVERLAP_FRACTION = 0.10


# Avoid concentrating all carriers in one task.
MAX_SLOTS_PER_TASK = 2


# ============================================================
# Load
# ============================================================

candidate_df = pd.read_csv(
    CANDIDATE_FILE
)

summary_df = pd.read_csv(
    MATCH_SUMMARY_FILE
)

pairs_df = pd.read_csv(
    PAIR_FILE
)


print("=" * 80)

print(
    "Candidate families:",
    len(candidate_df)
)

print(
    "Matched pairs:",
    len(pairs_df)
)

print("=" * 80)


# ============================================================
# Step 1
# Pair-level filtering
# ============================================================

pair_quality_mask = (

    (pairs_df["match_distance"]
     <= MAX_MATCH_DISTANCE)

    &

    (pairs_df["ee_pos_distance"]
     <= MAX_EE_POS_DISTANCE)

    &

    (pairs_df["ee_ori_distance"]
     <= MAX_EE_ORI_DISTANCE)

    &

    (pairs_df["joint_distance"]
     <= MAX_JOINT_DISTANCE)

    &

    (pairs_df["branch_fraction_diff"]
     <= MAX_BRANCH_FRACTION_DIFF)

    &

    (pairs_df["context_progress_diff"]
     <= MAX_CONTEXT_PROGRESS_DIFF)

    &

    (pairs_df["trajectory_length_diff"]
     <= MAX_TRAJECTORY_LENGTH_DIFF)

    &

    (
        pairs_df[
            "a_transition_occurrences"
        ]
        ==
        1
    )

    &

    (
        pairs_df[
            "b_transition_occurrences"
        ]
        ==
        1
    )

    &

    (
        pairs_df[
            "a_reward_max"
        ]
        >=
        1
    )

    &

    (
        pairs_df[
            "b_reward_max"
        ]
        >=
        1
    )

    &

    (
        pairs_df[
            "a_done_last"
        ]
        >=
        1
    )

    &

    (
        pairs_df[
            "b_done_last"
        ]
        >=
        1
    )
)


pairs_df[
    "passes_pair_filter"
] = pair_quality_mask


good_pairs = (
    pairs_df[
        pairs_df[
            "passes_pair_filter"
        ]
    ]
    .copy()
)


print(
    "\nHigh-quality matched pairs:"
)

print(
    len(good_pairs)
)


# ============================================================
# Step 2
# Count good pairs per candidate
# ============================================================

good_pair_counts = (

    good_pairs

    .groupby(
        "candidate_id"
    )

    .size()

    .rename(
        "good_pair_count"
    )

    .reset_index()
)


candidate_meta = (

    candidate_df

    .merge(
        summary_df,
        on=[
            "candidate_id",
            "task",
            "context",
            "source_primitive",
            "version_a_successor",
            "version_b_successor",
        ],
        how="inner",
    )

    .merge(
        good_pair_counts,
        on="candidate_id",
        how="left",
    )
)


candidate_meta[
    "good_pair_count"
] = (
    candidate_meta[
        "good_pair_count"
    ]
    .fillna(0)
    .astype(int)
)


candidate_meta[
    "good_pair_fraction"
] = (

    candidate_meta[
        "good_pair_count"
    ]

    /

    candidate_meta[
        "num_pairs"
    ]
)


# ============================================================
# Step 3
# Candidate-family filtering
# ============================================================

candidate_meta[
    "passes_family_filter"
] = (

    (
        candidate_meta[
            "good_pair_count"
        ]
        >=
        MIN_GOOD_PAIRS
    )

    &

    (
        candidate_meta[
            "matching_gain"
        ]
        >=
        MIN_MATCHING_GAIN
    )

    &

    (
        candidate_meta[
            "balance_score"
        ]
        >=
        MIN_BALANCE_SCORE
    )

    &

    (
        candidate_meta[
            "clean_coverage"
        ]
        >=
        MIN_CLEAN_COVERAGE
    )

    &

    (
        candidate_meta[
            "overlap_fraction"
        ]
        <=
        MAX_OVERLAP_FRACTION
    )
)


eligible = (

    candidate_meta[
        candidate_meta[
            "passes_family_filter"
        ]
    ]

    .copy()
)


# We deliberately do NOT create a mysterious weighted score.
#
# Priority:
#   1. more good pairs
#   2. stronger state-matching gain
#   3. better original candidate score
#
eligible = eligible.sort_values(

    [
        "good_pair_count",
        "matching_gain",
        "candidate_score",
    ],

    ascending=[
        False,
        False,
        False,
    ],
)


print(
    "\nEligible candidate families:"
)

print(
    len(eligible)
)


print(
    eligible[
        [
            "candidate_id",
            "context",
            "good_pair_count",
            "matching_gain",
            "balance_score",
            "clean_coverage",
        ]
    ]
    .round(3)
    .to_string(
        index=False
    )
)


# ============================================================
# Step 4
# Select matched bundles
#
# Important:
#
# A demonstration cannot be reused in two slots
# from the SAME task.
#
# ============================================================

used_demo_keys = set()

task_slot_counts = {}

selected_slots = []

selected_pairs = []

audit_rows = []


slot_number = 1


for _, candidate in eligible.iterrows():

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


    # --------------------------------------------------------
    # Limit number of slots from same task
    # --------------------------------------------------------

    current_task_slots = (
        task_slot_counts.get(
            task,
            0
        )
    )


    if (
        current_task_slots
        >=
        MAX_SLOTS_PER_TASK
    ):

        audit_rows.append(
            {
                "candidate_id":
                    candidate_id,

                "status":
                    "SKIPPED_TASK_LIMIT",

                "selected_pairs":
                    0,
            }
        )

        continue


    candidate_pairs = (

        good_pairs[
            good_pairs[
                "candidate_id"
            ]
            ==
            candidate_id
        ]

        .sort_values(
            [
                "match_distance",
                "ee_pos_distance",
            ],
            ascending=True,
        )
    )


    chosen = []


    for _, pair in candidate_pairs.iterrows():

        a_key = (
            task,
            pair[
                "a_demo"
            ]
        )

        b_key = (
            task,
            pair[
                "b_demo"
            ]
        )


        # ----------------------------------------------------
        # Cross-slot trajectory disjointness
        # ----------------------------------------------------

        if (
            a_key
            in
            used_demo_keys
        ):

            continue


        if (
            b_key
            in
            used_demo_keys
        ):

            continue


        chosen.append(
            pair
        )


        if (
            len(chosen)
            ==
            BUNDLE_SIZE
        ):

            break


    # --------------------------------------------------------
    # Candidate cannot form a full equal-size bundle
    # --------------------------------------------------------

    if (
        len(chosen)
        <
        BUNDLE_SIZE
    ):

        audit_rows.append(
            {
                "candidate_id":
                    candidate_id,

                "status":
                    "SKIPPED_NOT_ENOUGH_DISJOINT_PAIRS",

                "selected_pairs":
                    len(chosen),
            }
        )

        continue


    # ========================================================
    # Accept slot
    # ========================================================

    slot_id = (
        f"S{slot_number:02d}"
    )


    for pair_index, pair in enumerate(
        chosen,
        start=1,
    ):

        a_key = (
            task,
            pair[
                "a_demo"
            ]
        )

        b_key = (
            task,
            pair[
                "b_demo"
            ]
        )


        used_demo_keys.add(
            a_key
        )

        used_demo_keys.add(
            b_key
        )


        pair_record = (
            pair.to_dict()
        )


        pair_record[
            "slot_id"
        ] = slot_id


        pair_record[
            "slot_pair_index"
        ] = pair_index


        # We now freeze:
        #
        # A -> Version 0
        # B -> Version 1
        #
        pair_record[
            "version_0_demo"
        ] = pair[
            "a_demo"
        ]

        pair_record[
            "version_1_demo"
        ] = pair[
            "b_demo"
        ]


        selected_pairs.append(
            pair_record
        )


    chosen_df = pd.DataFrame(
        chosen
    )


    slot_record = {

        "slot_id":
            slot_id,

        "candidate_id":
            candidate_id,

        "task":
            task,

        "context":
            candidate[
                "context"
            ],

        "source_primitive":
            int(
                candidate[
                    "source_primitive"
                ]
            ),

        "version_0_successor":
            int(
                candidate[
                    "version_a_successor"
                ]
            ),

        "version_1_successor":
            int(
                candidate[
                    "version_b_successor"
                ]
            ),

        "bundle_size":
            BUNDLE_SIZE,

        "balance_score":
            candidate[
                "balance_score"
            ],

        "clean_coverage":
            candidate[
                "clean_coverage"
            ],

        "matching_gain":
            candidate[
                "matching_gain"
            ],

        "mean_selected_match_distance":
            chosen_df[
                "match_distance"
            ].mean(),

        "max_selected_match_distance":
            chosen_df[
                "match_distance"
            ].max(),

        "mean_selected_ee_pos_distance":
            chosen_df[
                "ee_pos_distance"
            ].mean(),

        "mean_selected_joint_distance":
            chosen_df[
                "joint_distance"
            ].mean(),
    }


    selected_slots.append(
        slot_record
    )


    task_slot_counts[
        task
    ] = (
        current_task_slots
        +
        1
    )


    audit_rows.append(
        {
            "candidate_id":
                candidate_id,

            "status":
                "SELECTED",

            "selected_pairs":
                BUNDLE_SIZE,

            "slot_id":
                slot_id,
        }
    )


    slot_number += 1


# ============================================================
# Step 5
# Save outputs
# ============================================================

final_slots_df = pd.DataFrame(
    selected_slots
)

final_pairs_df = pd.DataFrame(
    selected_pairs
)

audit_df = pd.DataFrame(
    audit_rows
)


final_slots_path = (
    OUTPUT_DIR
    / "final_slots.csv"
)

final_pairs_path = (
    OUTPUT_DIR
    / "final_slot_pairs.csv"
)

audit_path = (
    OUTPUT_DIR
    / "slot_selection_audit.csv"
)


final_slots_df.to_csv(
    final_slots_path,
    index=False
)

final_pairs_df.to_csv(
    final_pairs_path,
    index=False
)

audit_df.to_csv(
    audit_path,
    index=False
)


# ============================================================
# Step 6
# JSON manifest
# ============================================================

manifest = {

    "configuration": {

        "bundle_size":
            BUNDLE_SIZE,

        "max_slots_per_task":
            MAX_SLOTS_PER_TASK,

        "pair_thresholds": {

            "match_distance":
                MAX_MATCH_DISTANCE,

            "ee_pos_distance":
                MAX_EE_POS_DISTANCE,

            "ee_ori_distance":
                MAX_EE_ORI_DISTANCE,

            "joint_distance":
                MAX_JOINT_DISTANCE,

            "branch_fraction_diff":
                MAX_BRANCH_FRACTION_DIFF,

            "context_progress_diff":
                MAX_CONTEXT_PROGRESS_DIFF,

            "trajectory_length_diff":
                MAX_TRAJECTORY_LENGTH_DIFF,
        },

        "family_thresholds": {

            "min_good_pairs":
                MIN_GOOD_PAIRS,

            "min_matching_gain":
                MIN_MATCHING_GAIN,

            "min_balance_score":
                MIN_BALANCE_SCORE,

            "min_clean_coverage":
                MIN_CLEAN_COVERAGE,

            "max_overlap_fraction":
                MAX_OVERLAP_FRACTION,
        },
    },

    "slots": [],
}


for _, slot in final_slots_df.iterrows():

    slot_id = (
        slot[
            "slot_id"
        ]
    )


    pair_subset = (

        final_pairs_df[
            final_pairs_df[
                "slot_id"
            ]
            ==
            slot_id
        ]

        .sort_values(
            "slot_pair_index"
        )
    )


    slot_manifest = {

        "slot_id":
            slot_id,

        "candidate_id":
            slot[
                "candidate_id"
            ],

        "task":
            slot[
                "task"
            ],

        "context":
            slot[
                "context"
            ],

        "source_primitive":
            int(
                slot[
                    "source_primitive"
                ]
            ),

        "version_0_successor":
            int(
                slot[
                    "version_0_successor"
                ]
            ),

        "version_1_successor":
            int(
                slot[
                    "version_1_successor"
                ]
            ),

        "version_0_demos":
            pair_subset[
                "version_0_demo"
            ].tolist(),

        "version_1_demos":
            pair_subset[
                "version_1_demo"
            ].tolist(),

        "pairs": [],
    }


    for _, pair in pair_subset.iterrows():

        slot_manifest[
            "pairs"
        ].append(
            {

                "pair_index":
                    int(
                        pair[
                            "slot_pair_index"
                        ]
                    ),

                "version_0_demo":
                    pair[
                        "version_0_demo"
                    ],

                "version_1_demo":
                    pair[
                        "version_1_demo"
                    ],

                "match_distance":
                    float(
                        pair[
                            "match_distance"
                        ]
                    ),

                "ee_pos_distance":
                    float(
                        pair[
                            "ee_pos_distance"
                        ]
                    ),

                "joint_distance":
                    float(
                        pair[
                            "joint_distance"
                        ]
                    ),
            }
        )


    manifest[
        "slots"
    ].append(
        slot_manifest
    )


manifest_path = (
    OUTPUT_DIR
    / "final_slot_manifest.json"
)


with open(
    manifest_path,
    "w"
) as f:

    json.dump(
        manifest,
        f,
        indent=2
    )


# ============================================================
# Step 7
# Final overlap audit
# ============================================================

demo_usage = []


for _, pair in final_pairs_df.iterrows():

    demo_usage.append(
        {
            "slot_id":
                pair["slot_id"],

            "task":
                pair["task"],

            "demo":
                pair[
                    "version_0_demo"
                ],

            "version":
                0,
        }
    )

    demo_usage.append(
        {
            "slot_id":
                pair["slot_id"],

            "task":
                pair["task"],

            "demo":
                pair[
                    "version_1_demo"
                ],

            "version":
                1,
        }
    )


usage_df = pd.DataFrame(
    demo_usage
)


duplicates = (

    usage_df

    .groupby(
        [
            "task",
            "demo"
        ]
    )

    .filter(
        lambda x:
            len(x)
            >
            1
    )
)


overlap_path = (
    OUTPUT_DIR
    / "slot_demo_overlap_audit.csv"
)


duplicates.to_csv(
    overlap_path,
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
    "Final Slot Construction"
)

print(
    "=" * 100
)


print(
    "\nSelected slots:"
)

print(
    len(final_slots_df)
)


if len(final_slots_df) > 0:

    print(
        final_slots_df[
            [
                "slot_id",
                "candidate_id",
                "context",
                "source_primitive",
                "version_0_successor",
                "version_1_successor",
                "bundle_size",
                "matching_gain",
                "mean_selected_match_distance",
            ]
        ]
        .round(3)
        .to_string(
            index=False
        )
    )


print(
    "\nCross-slot duplicate trajectory usages:"
)

print(
    len(duplicates)
)


print(
    "\nSaved:"
)

print(
    final_slots_path
)

print(
    final_pairs_path
)

print(
    manifest_path
)

print(
    audit_path
)

print(
    overlap_path
)