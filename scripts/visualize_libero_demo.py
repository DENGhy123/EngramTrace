from pathlib import Path
import h5py
import numpy as np
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = (
    PROJECT_ROOT
    / "data"
    / "libero"
    / "libero_spatial"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# 1. Load one task file
# ---------------------------------------------------------

files = sorted(DATA_ROOT.glob("*.hdf5"))

if not files:
    raise FileNotFoundError("No LIBERO HDF5 files found.")

file_path = files[0]

with h5py.File(file_path, "r") as f:

    demos = sorted(
        f["data"].keys(),
        key=lambda x: int(x.split("_")[1])
    )

    demo_name = demos[0]
    demo = f["data"][demo_name]

    actions = np.asarray(demo["actions"])

    ee_pos = np.asarray(
        demo["obs"]["ee_pos"]
    )

    ee_ori = np.asarray(
        demo["obs"]["ee_ori"]
    )

    gripper_states = np.asarray(
        demo["obs"]["gripper_states"]
    )

    agentview = np.asarray(
        demo["obs"]["agentview_rgb"]
    )

    eye_in_hand = np.asarray(
        demo["obs"]["eye_in_hand_rgb"]
    )

    language_instruction = (
        f["data"]
        .attrs["problem_info"]
    )


print("File:", file_path.name)
print("Demo:", demo_name)
print("Steps:", len(actions))
print("Actions:", actions.shape)
print("EE position:", ee_pos.shape)
print("EE orientation:", ee_ori.shape)
print("Gripper states:", gripper_states.shape)


# ---------------------------------------------------------
# 2. Plot translation actions
# ---------------------------------------------------------

time = np.arange(len(actions))

plt.figure(figsize=(12, 6))

labels = ["dx", "dy", "dz"]

for i in range(3):
    plt.plot(
        time,
        actions[:, i],
        label=labels[i]
    )

plt.axhline(
    0,
    linewidth=0.8
)

plt.xlabel("Time step")
plt.ylabel("Normalized command")
plt.title(
    "LIBERO Translation Actions"
)

plt.legend()
plt.tight_layout()

path = OUTPUT_DIR / "demo0_translation_actions.png"

plt.savefig(
    path,
    dpi=200
)

plt.close()

print("Saved:", path)


# ---------------------------------------------------------
# 3. Plot rotation actions
# ---------------------------------------------------------

plt.figure(figsize=(12, 6))

labels = ["dRx", "dRy", "dRz"]

for i in range(3):
    plt.plot(
        time,
        actions[:, i + 3],
        label=labels[i]
    )

plt.axhline(
    0,
    linewidth=0.8
)

plt.xlabel("Time step")
plt.ylabel("Normalized command")
plt.title(
    "LIBERO Orientation Actions"
)

plt.legend()
plt.tight_layout()

path = OUTPUT_DIR / "demo0_rotation_actions.png"

plt.savefig(
    path,
    dpi=200
)

plt.close()

print("Saved:", path)


# ---------------------------------------------------------
# 4. Plot gripper command
# ---------------------------------------------------------

plt.figure(figsize=(12, 4))

plt.step(
    time,
    actions[:, 6],
    where="post"
)

plt.xlabel("Time step")
plt.ylabel("Gripper command")

plt.yticks(
    [-1, 1],
    ["Open (-1)", "Close (+1)"]
)

plt.title(
    "LIBERO Gripper Action"
)

plt.tight_layout()

path = OUTPUT_DIR / "demo0_gripper_action.png"

plt.savefig(
    path,
    dpi=200
)

plt.close()

print("Saved:", path)


# ---------------------------------------------------------
# 5. Plot real end-effector XYZ position
# ---------------------------------------------------------

plt.figure(figsize=(12, 6))

labels = ["x", "y", "z"]

for i in range(3):
    plt.plot(
        time,
        ee_pos[:, i],
        label=labels[i]
    )

plt.xlabel("Time step")
plt.ylabel("EE position")
plt.title(
    "End-Effector Position"
)

plt.legend()
plt.tight_layout()

path = OUTPUT_DIR / "demo0_ee_position.png"

plt.savefig(
    path,
    dpi=200
)

plt.close()

print("Saved:", path)


# ---------------------------------------------------------
# 6. Plot 3D trajectory
# ---------------------------------------------------------

fig = plt.figure(figsize=(8, 7))

ax = fig.add_subplot(
    111,
    projection="3d"
)

ax.plot(
    ee_pos[:, 0],
    ee_pos[:, 1],
    ee_pos[:, 2]
)

ax.scatter(
    ee_pos[0, 0],
    ee_pos[0, 1],
    ee_pos[0, 2],
    s=60,
    label="Start"
)

ax.scatter(
    ee_pos[-1, 0],
    ee_pos[-1, 1],
    ee_pos[-1, 2],
    s=60,
    label="End"
)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

ax.set_title(
    "End-Effector 3D Trajectory"
)

ax.legend()

plt.tight_layout()

path = OUTPUT_DIR / "demo0_ee_trajectory_3d.png"

plt.savefig(
    path,
    dpi=200
)

plt.close()

print("Saved:", path)


# ---------------------------------------------------------
# 7. Extract representative frames
# ---------------------------------------------------------

frame_indices = np.linspace(
    0,
    len(agentview) - 1,
    8,
    dtype=int
)

fig, axes = plt.subplots(
    2,
    4,
    figsize=(14, 7)
)

for ax, idx in zip(
    axes.flat,
    frame_indices
):

    ax.imshow(
        np.flipud(
            agentview[idx]
        )
    )

    ax.set_title(
        f"t={idx}"
    )

    ax.axis("off")

plt.suptitle(
    "Agent View - Representative Frames"
)

plt.tight_layout()

path = OUTPUT_DIR / "demo0_agentview_frames.png"

plt.savefig(
    path,
    dpi=200
)

plt.close()

print("Saved:", path)


# ---------------------------------------------------------
# 8. Detect gripper switching times
# ---------------------------------------------------------

gripper = actions[:, 6]

switches = np.where(
    np.diff(gripper) != 0
)[0] + 1

print("\nGripper switch time steps:")
print(switches)

for t in switches:

    print(
        f"t={t}: "
        f"{gripper[t-1]} -> {gripper[t]}"
    )