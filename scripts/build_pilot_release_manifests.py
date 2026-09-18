from pathlib import Path
import json

import h5py
import pandas as pd


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

FINAL_SLOT_MANIFEST = (
    PROJECT_ROOT
    / "outputs"
    / "final_slots"
    / "final_slot_manifest.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "pilot_releases"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# Configuration
# ============================================================

# First 1-bit acquisition experiments.
#
# S02/S05 = primary
# S07      = independent replication
#
PILOT_SLOTS = [
    "S02",
    "S05",
    "S07",
]


# Use 4 of 5 matched pairs for training.
ACTIVE_PAIRS = 4


# The remaining one matched pair is kept fully held out.
HELDOUT_PAIRS = 1


# ============================================================
# Helpers
# ============================================================

def demo_sort_key(name):

    return int(
        name.split("_")[1]
    )


def task_to_hdf5(task):

    path = (
        DATA_ROOT
        /
        f"{task}_demo.hdf5"
    )

    if not path.exists():

        raise FileNotFoundError(
            f"Cannot find:\n{path}"
        )

    return path


# ============================================================
# Load slot manifest
# ============================================================

with open(
    FINAL_SLOT_MANIFEST,
    "r",
) as f:

    manifest = json.load(f)


slot_map = {

    slot["slot_id"]:
        slot

    for slot in manifest["slots"]
}


# ============================================================
# Build releases
# ============================================================

release_manifest = {

    "description":
        (
            "Pilot 1-bit EngramTrace releases. "
            "All demonstrations are intact; "
            "only trajectory selection differs."
        ),

    "active_pairs_per_slot":
        ACTIVE_PAIRS,

    "heldout_pairs_per_slot":
        HELDOUT_PAIRS,

    "slots": [],
}


csv_rows = []


for slot_id in PILOT_SLOTS:

    if slot_id not in slot_map:

        raise KeyError(
            f"Cannot find slot {slot_id}"
        )


    slot = slot_map[
        slot_id
    ]


    task = slot[
        "task"
    ]


    hdf5_path = task_to_hdf5(
        task
    )


    # --------------------------------------------------------
    # Read all 50 demo IDs from original task.
    # --------------------------------------------------------

    with h5py.File(
        hdf5_path,
        "r",
    ) as f:

        all_demos = sorted(
            f["data"].keys(),
            key=demo_sort_key,
        )


    print(
        "\n" + "=" * 80
    )

    print(
        f"{slot_id}: {task}"
    )

    print(
        "Original demos:",
        len(all_demos)
    )


    # --------------------------------------------------------
    # Matched pairs
    #
    # They were already stored in quality order.
    # --------------------------------------------------------

    pairs = sorted(
        slot["pairs"],
        key=lambda x:
            x["pair_index"],
    )


    if len(pairs) != 5:

        raise RuntimeError(
            f"{slot_id} expected 5 pairs, "
            f"got {len(pairs)}"
        )


    active = (
        pairs[
            :ACTIVE_PAIRS
        ]
    )

    heldout = (
        pairs[
            ACTIVE_PAIRS:
        ]
    )


    active_v0 = [

        x[
            "version_0_demo"
        ]

        for x in active
    ]


    active_v1 = [

        x[
            "version_1_demo"
        ]

        for x in active
    ]


    heldout_v0 = [

        x[
            "version_0_demo"
        ]

        for x in heldout
    ]


    heldout_v1 = [

        x[
            "version_1_demo"
        ]

        for x in heldout
    ]


    # --------------------------------------------------------
    # Remove ALL ten slot-related demos from common pool.
    #
    # This includes the held-out pair.
    # --------------------------------------------------------

    all_slot_demos = set(
        slot[
            "version_0_demos"
        ]
        +
        slot[
            "version_1_demos"
        ]
    )


    common = [

        demo

        for demo in all_demos

        if demo
        not in
        all_slot_demos
    ]


    if len(common) != 40:

        raise RuntimeError(
            f"{slot_id}: expected 40 common demos, "
            f"got {len(common)}"
        )


    # ========================================================
    # Three primary release conditions
    # ========================================================

    version0 = (
        common
        +
        active_v0
    )


    version1 = (
        common
        +
        active_v1
    )


    # Balanced Neutral:
    #
    # pair 1,2 -> V0
    # pair 3,4 -> V1
    #
    neutral = (

        common

        +

        active_v0[:2]

        +

        active_v1[2:]
    )


    # Complementary neutral for later robustness.
    #
    neutral_complement = (

        common

        +

        active_v1[:2]

        +

        active_v0[2:]
    )


    releases = {

        "V0":
            sorted(
                version0,
                key=demo_sort_key,
            ),

        "NEUTRAL":
            sorted(
                neutral,
                key=demo_sort_key,
            ),

        "NEUTRAL_COMPLEMENT":
            sorted(
                neutral_complement,
                key=demo_sort_key,
            ),

        "V1":
            sorted(
                version1,
                key=demo_sort_key,
            ),
    }


    # ========================================================
    # Sanity checks
    # ========================================================

    for condition, demos in (
        releases.items()
    ):

        if len(demos) != 44:

            raise RuntimeError(
                f"{slot_id}/{condition}: "
                f"expected 44 demos, "
                f"got {len(demos)}"
            )


        if len(set(demos)) != 44:

            raise RuntimeError(
                f"{slot_id}/{condition}: "
                f"duplicate demos detected"
            )


    # Held-out pair must not occur in any release.
    heldout_set = set(
        heldout_v0
        +
        heldout_v1
    )


    for condition, demos in (
        releases.items()
    ):

        overlap = (
            heldout_set
            &
            set(demos)
        )

        if overlap:

            raise RuntimeError(
                f"Held-out leakage in "
                f"{slot_id}/{condition}: "
                f"{overlap}"
            )


    # Common 40 must occur everywhere.
    for condition, demos in (
        releases.items()
    ):

        if not set(common).issubset(
            set(demos)
        ):

            raise RuntimeError(
                f"{slot_id}/{condition}: "
                f"missing common demos"
            )


    # ========================================================
    # Store JSON
    # ========================================================

    slot_output = {

        "slot_id":
            slot_id,

        "task":
            task,

        "context":
            slot[
                "context"
            ],

        "source_primitive":
            slot[
                "source_primitive"
            ],

        "version_0_successor":
            slot[
                "version_0_successor"
            ],

        "version_1_successor":
            slot[
                "version_1_successor"
            ],

        "original_num_demos":
            len(all_demos),

        "common_num_demos":
            len(common),

        "release_num_demos":
            44,

        "carrier_num_demos":
            ACTIVE_PAIRS,

        "carrier_fraction":
            ACTIVE_PAIRS / 44.0,

        "common_demos":
            common,

        "active_version_0_demos":
            active_v0,

        "active_version_1_demos":
            active_v1,

        "heldout_version_0_demos":
            heldout_v0,

        "heldout_version_1_demos":
            heldout_v1,

        "releases":
            releases,
    }


    release_manifest[
        "slots"
    ].append(
        slot_output
    )


    # ========================================================
    # Store CSV rows
    # ========================================================

    for condition, demos in (
        releases.items()
    ):

        for demo in demos:

            if demo in common:

                role = "COMMON"

            elif demo in active_v0:

                role = "CARRIER_V0"

            elif demo in active_v1:

                role = "CARRIER_V1"

            else:

                role = "UNKNOWN"


            csv_rows.append(
                {

                    "slot_id":
                        slot_id,

                    "task":
                        task,

                    "context":
                        slot[
                            "context"
                        ],

                    "condition":
                        condition,

                    "demo":
                        demo,

                    "role":
                        role,
                }
            )


    print(
        "Common:",
        len(common)
    )

    print(
        "Active V0:",
        active_v0
    )

    print(
        "Active V1:",
        active_v1
    )

    print(
        "Held-out V0:",
        heldout_v0
    )

    print(
        "Held-out V1:",
        heldout_v1
    )

    print(
        "Each release:",
        len(version0)
    )

    print(
        "Carrier fraction:",
        round(
            ACTIVE_PAIRS / 44.0,
            4
        )
    )


# ============================================================
# Save
# ============================================================

json_path = (
    OUTPUT_DIR
    / "pilot_release_manifest.json"
)


with open(
    json_path,
    "w",
) as f:

    json.dump(
        release_manifest,
        f,
        indent=2,
    )


csv_df = pd.DataFrame(
    csv_rows
)


csv_path = (
    OUTPUT_DIR
    / "pilot_release_manifest.csv"
)


csv_df.to_csv(
    csv_path,
    index=False,
)


print(
    "\n"
    +
    "=" * 80
)

print(
    "Pilot release construction complete."
)

print(
    "=" * 80
)

print(
    "\nSaved:"
)

print(
    json_path
)

print(
    csv_path
)