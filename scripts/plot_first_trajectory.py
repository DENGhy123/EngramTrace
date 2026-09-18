from pathlib import Path
import h5py
import numpy as np
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "libero"
    / "libero_spatial"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


files = sorted(DATA_DIR.glob("*.hdf5"))
file_path = files[0]

with h5py.File(file_path, "r") as f:

    demos = sorted(f["data"].keys())

    demo = f["data"][demos[0]]

    actions = np.asarray(demo["actions"])


print("Action shape:", actions.shape)

plt.figure(figsize=(12, 6))

for dim in range(actions.shape[1]):
    plt.plot(
        actions[:, dim],
        label=f"action_{dim}",
        linewidth=1,
    )

plt.xlabel("Time step")
plt.ylabel("Action value")
plt.title("LIBERO: First Demonstration Action Trajectory")

plt.legend(
    ncol=2,
    fontsize=8,
)

plt.tight_layout()

output_path = (
    OUTPUT_DIR
    / "first_libero_action_trajectory.png"
)

plt.savefig(
    output_path,
    dpi=200,
)

print("Saved to:")
print(output_path)