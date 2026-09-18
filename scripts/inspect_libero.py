from pathlib import Path
import h5py
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "libero"


def print_hdf5_tree(group, prefix=""):
    for key in group.keys():
        item = group[key]

        if isinstance(item, h5py.Dataset):
            print(
                f"{prefix}{key}: "
                f"shape={item.shape}, dtype={item.dtype}"
            )

        elif isinstance(item, h5py.Group):
            print(f"{prefix}{key}/")
            print_hdf5_tree(item, prefix + "    ")


files = sorted(DATA_ROOT.rglob("*.hdf5"))

if not files:
    raise FileNotFoundError(
        f"No HDF5 files found under:\n{DATA_ROOT}"
    )

print(f"Found {len(files)} HDF5 files.")

file_path = files[0]

print("\n" + "=" * 80)
print("Opening file:")
print(file_path)
print("=" * 80)

with h5py.File(file_path, "r") as f:

    print("\n[1] Top-level keys")
    print(list(f.keys()))

    print("\n[2] Complete HDF5 structure")
    print_hdf5_tree(f)

    if "data" not in f:
        raise RuntimeError(
            "Expected a top-level 'data' group but did not find it."
        )

    data = f["data"]

    print("\n[3] data attributes")
    for key, value in data.attrs.items():
        print(f"{key}: {value}")

    demos = sorted(data.keys())

    print("\n[4] Number of demonstrations")
    print(len(demos))

    print("\n[5] First five demonstrations")
    print(demos[:5])

    first_demo_name = demos[0]
    demo = data[first_demo_name]

    print("\n" + "=" * 80)
    print("Inspecting:", first_demo_name)
    print("=" * 80)

    print("\n[6] Demo keys")
    print(list(demo.keys()))

    if "actions" in demo:
        actions = np.asarray(demo["actions"])

        print("\n[7] Actions")
        print("shape:", actions.shape)
        print("dtype:", actions.dtype)

        print("\nFirst 5 actions:")
        print(actions[:5])

        print("\nAction min:")
        print(actions.min(axis=0))

        print("\nAction max:")
        print(actions.max(axis=0))

        print("\nAction mean:")
        print(actions.mean(axis=0))

        print("\nAction std:")
        print(actions.std(axis=0))

    if "states" in demo:
        states = np.asarray(demo["states"])

        print("\n[8] States")
        print("shape:", states.shape)
        print("dtype:", states.dtype)

    if "robot_states" in demo:
        robot_states = np.asarray(demo["robot_states"])

        print("\n[9] Robot states")
        print("shape:", robot_states.shape)
        print("dtype:", robot_states.dtype)

    if "obs" in demo:
        print("\n[10] Observation keys")

        obs = demo["obs"]

        for key in obs.keys():
            arr = obs[key]

            print(
                f"{key:30s} "
                f"shape={arr.shape}, "
                f"dtype={arr.dtype}"
            )