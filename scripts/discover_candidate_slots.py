from pathlib import Path
from collections import defaultdict
from itertools import combinations
import json

import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

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
    / "candidate_slots"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ------------------------------------------------------------
# Candidate filtering thresholds
# ------------------------------------------------------------

# The source primitive must appear with some outgoing transition
# in at least this many unique demonstrations.
MIN_SOURCE_DEMOS = 10

# After removing ambiguous demonstrations,
# each A/B branch must have at least this many demos.
MIN_BRANCH_DEMOS = 5

# The smaller branch should occupy at least this fraction
# among clean A/B demonstrations.
MIN_MINOR_FRACTION = 0.20

# If many demos contain BOTH source->A and source->B,
# the branch is not a clean trajectory-level alternative.
MAX_OVERLAP_FRACTION = 0.25

# Fraction of source demonstrations that can be cleanly
# explained by A-only or B-only.
MIN_CLEAN_COVERAGE = 0.50


# ============================================================
# Helpers
# ============================================================

def task_name_from_file(task_file):
    """
    Convert:
        xxx_demo.hdf5
    to:
        xxx
    """

    name = str(task_file)

    if name.endswith("_demo.hdf5"):
        name = name[:-10]

    elif name.endswith(".hdf5"):
        name = name[:-5]

    return name


def relative_context_name(phase, relative_position):
    """
    Convert coarse macro phase into Context V1.

    PRE_GRASP and HOLDING are divided into thirds.

    POST_RELEASE remains one context because it is
    comparatively short in LIBERO-Spatial.
    """

    phase = str(phase)

    if phase in {
        "PRE_GRASP",
        "HOLDING",
    }:

        if relative_position < 1.0 / 3.0:
            suffix = "EARLY"

        elif relative_position < 2.0 / 3.0:
            suffix = "MIDDLE"

        else:
            suffix = "LATE"

        return f"{phase}_{suffix}"

    if phase == "POST_RELEASE":

        return "POST_RELEASE_ALL"

    # Fallback in case future datasets contain trajectories
    # where no grasp event is available.
    if phase == "FULL_TRAJECTORY":

        if relative_position < 1.0 / 3.0:
            return "FULL_EARLY"

        elif relative_position < 2.0 / 3.0:
            return "FULL_MIDDLE"

        return "FULL_LATE"

    return phase


def compress_sequence(sequence):
    """
    Run-length compression.

    Example:
        [2, 2, 2, 7, 7, 0]
    ->
        [2, 7, 0]

    This prevents long-lasting motion primitives
    from creating artificial P2->P2 transitions.
    """

    compressed = []

    previous = None

    for item in sequence:

        item = int(item)

        if item != previous:

            compressed.append(item)

            previous = item

    return compressed


def join_demo_names(demos):
    """
    Store demo IDs compactly inside CSV files.
    """

    return ";".join(
        sorted(
            demos,
            key=lambda x: int(
                x.split("_")[1]
            )
        )
    )


# ============================================================
# Load data
# ============================================================

if not INPUT_CSV.exists():

    raise FileNotFoundError(
        f"Cannot find:\n{INPUT_CSV}"
    )


df = pd.read_csv(
    INPUT_CSV
)


required_columns = {

    "task_file",
    "demo",
    "phase",
    "chunk_index",
    "start",
    "end",
    "primitive",
}


missing_columns = (
    required_columns
    -
    set(df.columns)
)


if missing_columns:

    raise RuntimeError(
        "Missing columns: "
        + str(missing_columns)
    )


print("=" * 80)

print(
    "Loaded motion chunks:"
)

print(
    len(df)
)

print(
    "Tasks:",
    df["task_file"].nunique()
)

print(
    "Demonstrations:",
    df[
        [
            "task_file",
            "demo"
        ]
    ]
    .drop_duplicates()
    .shape[0]
)

print("=" * 80)


df["task"] = (
    df["task_file"]
    .apply(task_name_from_file)
)


# ============================================================
# Step 1
# Assign Context V1
# ============================================================

df["context"] = None

df["relative_phase_position"] = np.nan


group_columns = [

    "task_file",
    "demo",
    "phase",
]


for group_key, group in df.groupby(
    group_columns,
    sort=False
):

    group = group.sort_values(
        "start"
    )

    phase_start = (
        group["start"].min()
    )

    phase_end = (
        group["end"].max()
    )

    duration = (
        phase_end
        -
        phase_start
    )

    if duration <= 0:
        duration = 1


    for idx, row in group.iterrows():

        midpoint = (
            row["start"]
            +
            row["end"]
        ) / 2.0

        relative_position = (
            midpoint
            -
            phase_start
        ) / duration

        relative_position = float(
            np.clip(
                relative_position,
                0.0,
                1.0,
            )
        )

        context = relative_context_name(
            row["phase"],
            relative_position,
        )

        df.loc[
            idx,
            "relative_phase_position"
        ] = relative_position

        df.loc[
            idx,
            "context"
        ] = context


contextualized_path = (
    OUTPUT_DIR
    / "contextualized_motion_chunks.csv"
)


df.to_csv(
    contextualized_path,
    index=False
)


print(
    "\nSaved contextualized chunks:"
)

print(
    contextualized_path
)


# ============================================================
# Step 2
# Construct context-specific primitive sequences
# ============================================================

context_sequences = {}


# Mapping:
#
# (task, context, source, successor)
#     -> set(demo IDs)
#
transition_demos = defaultdict(
    set
)


# Mapping:
#
# (task, context, source)
#     -> demos containing ANY outgoing transition
#
source_demos = defaultdict(
    set
)


sequence_group_columns = [

    "task",
    "demo",
    "context",
]


for (
    task,
    demo,
    context
), group in df.groupby(
    sequence_group_columns,
    sort=False
):

    group = group.sort_values(
        [
            "start",
            "chunk_index",
        ]
    )

    raw_sequence = (
        group["primitive"]
        .astype(int)
        .tolist()
    )


    compressed_sequence = (
        compress_sequence(
            raw_sequence
        )
    )


    sequence_key = (
        f"{task}::{demo}::{context}"
    )


    context_sequences[
        sequence_key
    ] = {

        "task":
            task,

        "demo":
            demo,

        "context":
            context,

        "raw_sequence":
            raw_sequence,

        "compressed_sequence":
            compressed_sequence,
    }


    if len(
        compressed_sequence
    ) < 2:

        continue


    transitions = list(
        zip(
            compressed_sequence[:-1],
            compressed_sequence[1:],
        )
    )


    # --------------------------------------------------------
    # VERY IMPORTANT:
    #
    # Count one transition at most once per demonstration.
    #
    # Frames / chunks inside one demo are NOT treated as
    # independent samples.
    # --------------------------------------------------------

    unique_transitions = set(
        transitions
    )


    for source, successor in unique_transitions:

        transition_demos[
            (
                task,
                context,
                int(source),
                int(successor),
            )
        ].add(
            demo
        )


        source_demos[
            (
                task,
                context,
                int(source),
            )
        ].add(
            demo
        )


sequence_path = (
    OUTPUT_DIR
    / "context_primitive_sequences.json"
)


with open(
    sequence_path,
    "w"
) as f:

    json.dump(
        context_sequences,
        f,
        indent=2,
    )


print(
    "\nSaved context sequences:"
)

print(
    sequence_path
)


# ============================================================
# Step 3
# Build full transition inventory
# ============================================================

transition_rows = []


for (
    task,
    context,
    source,
    successor
), demos in transition_demos.items():

    transition_rows.append(
        {

            "task":
                task,

            "context":
                context,

            "source_primitive":
                source,

            "successor_primitive":
                successor,

            "demo_count":
                len(demos),

            "demos":
                join_demo_names(
                    demos
                ),
        }
    )


transition_df = pd.DataFrame(
    transition_rows
)


if len(transition_df) > 0:

    transition_df = (
        transition_df
        .sort_values(
            [
                "task",
                "context",
                "source_primitive",
                "demo_count",
            ],
            ascending=[
                True,
                True,
                True,
                False,
            ],
        )
        .reset_index(
            drop=True
        )
    )


transition_path = (
    OUTPUT_DIR
    / "transition_inventory.csv"
)


transition_df.to_csv(
    transition_path,
    index=False
)


print(
    "\nSaved transition inventory:"
)

print(
    transition_path
)


# ============================================================
# Step 4
# Build successor map
# ============================================================

successor_map = defaultdict(
    dict
)


for (
    task,
    context,
    source,
    successor
), demos in transition_demos.items():

    successor_map[
        (
            task,
            context,
            source,
        )
    ][successor] = set(
        demos
    )


# ============================================================
# Step 5
# Discover candidate A/B branch pairs
# ============================================================

candidate_rows = []


for (
    task,
    context,
    source
), successors in successor_map.items():

    all_source_demos = (
        source_demos[
            (
                task,
                context,
                source,
            )
        ]
    )

    source_demo_count = len(
        all_source_demos
    )


    if (
        source_demo_count
        <
        MIN_SOURCE_DEMOS
    ):

        continue


    successor_ids = sorted(
        successors.keys()
    )


    if len(
        successor_ids
    ) < 2:

        continue


    # --------------------------------------------------------
    # Evaluate ALL successor pairs rather than simply taking
    # the two most frequent ones.
    #
    # This is more robust when the top two overlap strongly.
    # --------------------------------------------------------

    for (
        successor_a,
        successor_b
    ) in combinations(
        successor_ids,
        2
    ):

        demos_a = set(
            successors[
                successor_a
            ]
        )

        demos_b = set(
            successors[
                successor_b
            ]
        )


        # Demos that exhibit BOTH branches.
        overlap = (
            demos_a
            &
            demos_b
        )


        # Clean trajectory-level versions.
        #
        # These are the demos we may later use as candidate
        # Version A and Version B inventories.
        demos_a_only = (
            demos_a
            -
            demos_b
        )

        demos_b_only = (
            demos_b
            -
            demos_a
        )


        n_a_total = len(
            demos_a
        )

        n_b_total = len(
            demos_b
        )

        n_overlap = len(
            overlap
        )

        n_a_clean = len(
            demos_a_only
        )

        n_b_clean = len(
            demos_b_only
        )


        clean_total = (
            n_a_clean
            +
            n_b_clean
        )


        pair_union = (
            demos_a
            |
            demos_b
        )

        pair_union_count = len(
            pair_union
        )


        if clean_total == 0:

            continue


        # ----------------------------------------------------
        # Balance
        #
        # 1.0 = perfectly balanced
        # ----------------------------------------------------

        minor_fraction = (
            min(
                n_a_clean,
                n_b_clean
            )
            /
            clean_total
        )


        balance_score = (
            2.0
            *
            minor_fraction
        )


        # ----------------------------------------------------
        # How many source demos are represented by this pair?
        # ----------------------------------------------------

        pair_coverage = (
            pair_union_count
            /
            source_demo_count
        )


        clean_coverage = (
            clean_total
            /
            source_demo_count
        )


        # ----------------------------------------------------
        # Ambiguity
        # ----------------------------------------------------

        overlap_fraction = (
            n_overlap
            /
            pair_union_count
            if pair_union_count > 0
            else 0.0
        )


        # ----------------------------------------------------
        # Demos explained by other successor branches
        # ----------------------------------------------------

        other_demos = (
            all_source_demos
            -
            pair_union
        )


        # ----------------------------------------------------
        # Candidate ranking heuristic
        #
        # IMPORTANT:
        # This is NOT a security score.
        #
        # Higher when:
        #   - A/B are balanced
        #   - many source demos are cleanly represented
        #   - little overlap exists
        #   - absolute support is larger
        # ----------------------------------------------------

        candidate_score = (

            balance_score

            *

            clean_coverage

            *

            (
                1.0
                -
                overlap_fraction
            )

            *

            np.log1p(
                clean_total
            )
        )


        # ----------------------------------------------------
        # Filtering
        # ----------------------------------------------------

        passes = (

            n_a_clean
            >=
            MIN_BRANCH_DEMOS

            and

            n_b_clean
            >=
            MIN_BRANCH_DEMOS

            and

            minor_fraction
            >=
            MIN_MINOR_FRACTION

            and

            overlap_fraction
            <=
            MAX_OVERLAP_FRACTION

            and

            clean_coverage
            >=
            MIN_CLEAN_COVERAGE
        )


        candidate_rows.append(
            {

                "task":
                    task,

                "context":
                    context,

                "source_primitive":
                    int(source),

                "version_a_successor":
                    int(successor_a),

                "version_b_successor":
                    int(successor_b),

                "source_demo_count":
                    source_demo_count,

                "a_total_demo_count":
                    n_a_total,

                "b_total_demo_count":
                    n_b_total,

                "a_clean_demo_count":
                    n_a_clean,

                "b_clean_demo_count":
                    n_b_clean,

                "overlap_demo_count":
                    n_overlap,

                "other_demo_count":
                    len(
                        other_demos
                    ),

                "minor_fraction":
                    minor_fraction,

                "balance_score":
                    balance_score,

                "pair_coverage":
                    pair_coverage,

                "clean_coverage":
                    clean_coverage,

                "overlap_fraction":
                    overlap_fraction,

                "candidate_score":
                    candidate_score,

                "passes_filter":
                    passes,

                "version_a_demos":
                    join_demo_names(
                        demos_a_only
                    ),

                "version_b_demos":
                    join_demo_names(
                        demos_b_only
                    ),

                "ambiguous_demos":
                    join_demo_names(
                        overlap
                    ),

                "other_demos":
                    join_demo_names(
                        other_demos
                    ),
            }
        )


candidate_df = pd.DataFrame(
    candidate_rows
)


if len(candidate_df) == 0:

    print(
        "\nNo candidate branch pairs found."
    )

    raise SystemExit


# ============================================================
# Step 6
# Sort and assign IDs
# ============================================================

candidate_df = (
    candidate_df
    .sort_values(
        "candidate_score",
        ascending=False
    )
    .reset_index(
        drop=True
    )
)


candidate_df.insert(
    0,
    "candidate_id",
    [
        f"C{i:04d}"
        for i in range(
            1,
            len(candidate_df) + 1
        )
    ]
)


all_candidate_path = (
    OUTPUT_DIR
    / "all_candidate_branches.csv"
)


candidate_df.to_csv(
    all_candidate_path,
    index=False
)


# ============================================================
# Step 7
# Keep filtered candidates
# ============================================================

selected_df = (
    candidate_df[
        candidate_df[
            "passes_filter"
        ]
        ==
        True
    ]
    .copy()
    .reset_index(
        drop=True
    )
)


selected_path = (
    OUTPUT_DIR
    / "candidate_slots.csv"
)


selected_df.to_csv(
    selected_path,
    index=False
)


print(
    "\nSaved all branch candidates:"
)

print(
    all_candidate_path
)


print(
    "\nSaved filtered candidate slots:"
)

print(
    selected_path
)


# ============================================================
# Step 8
# Context summary
# ============================================================

if len(selected_df) > 0:

    context_summary = (

        selected_df

        .groupby(
            "context"
        )

        .agg(

            num_candidates=
                (
                    "candidate_id",
                    "count"
                ),

            mean_clean_support=
                (
                    "source_demo_count",
                    "mean"
                ),

            mean_balance=
                (
                    "balance_score",
                    "mean"
                ),

            mean_clean_coverage=
                (
                    "clean_coverage",
                    "mean"
                ),
        )

        .reset_index()
    )


    context_summary_path = (
        OUTPUT_DIR
        / "candidate_context_summary.csv"
    )


    context_summary.to_csv(
        context_summary_path,
        index=False
    )


# ============================================================
# Step 9
# Print results
# ============================================================

print(
    "\n"
    +
    "=" * 100
)

print(
    "Candidate Slot Discovery Summary"
)

print(
    "=" * 100
)


print(
    "\nAll branch pairs evaluated:"
)

print(
    len(candidate_df)
)


print(
    "\nCandidates passing filters:"
)

print(
    len(selected_df)
)


print(
    "\nNumber of tasks with at least one candidate:"
)

print(
    selected_df["task"].nunique()
    if len(selected_df) > 0
    else 0
)


print(
    "\nCandidate counts by context:"
)


if len(selected_df) > 0:

    print(
        selected_df[
            "context"
        ]
        .value_counts()
    )


# ============================================================
# Top candidates
# ============================================================

print(
    "\n"
    +
    "=" * 100
)

print(
    "Top Candidate Slots"
)

print(
    "=" * 100
)


if len(selected_df) == 0:

    print(
        "No candidates passed the current thresholds."
    )

else:

    display_columns = [

        "candidate_id",

        "task",

        "context",

        "source_primitive",

        "version_a_successor",

        "version_b_successor",

        "source_demo_count",

        "a_clean_demo_count",

        "b_clean_demo_count",

        "balance_score",

        "clean_coverage",

        "overlap_fraction",

        "candidate_score",
    ]


    print(
        selected_df[
            display_columns
        ]
        .head(30)
        .round(3)
        .to_string(
            index=False
        )
    )


print(
    "\nDone."
)