import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp


def plot_Fs(F):
    num_dynamics = F.shape[0]
    num_rows = (num_dynamics - 1) // 6 + 1
    num_cols = min(6, num_dynamics)
    maxabsval = np.max(np.abs(F))

    plt.figure(figsize=(num_cols, num_rows))
    for i in range(F.shape[0]):
        plt.subplot(num_rows, num_cols, i + 1)
        plt.imshow(F[i], aspect="auto", vmin=-maxabsval, vmax=maxabsval)
        plt.axis("off")

    plt.tight_layout()


def plot_X(X):
    num_latents = X.shape[0]
    num_timepoints = X.shape[1]

    plt.figure(figsize=(20, 8))
    for i in range(num_latents):
        plt.subplot(num_latents, 1, i + 1)
        plt.plot(np.arange(num_timepoints), X[i, :].T)

    plt.tight_layout()


def plot_C(C):
    plt.figure(figsize=(10, 2))
    plt.imshow(C, aspect="auto", interpolation="none")

    plt.tight_layout()

def plot_Cs(C, trial_ids: np.array = []):
    trial_keys = list(C.keys())
    num_trials = len(C)
    
    if len(trial_ids) == 0:
        num_rows = (num_trials - 1) // 6 + 1
        num_cols = min(6, num_trials)
        plt.figure(figsize=(num_cols, num_rows))
        i=0
        for k in trial_keys:
            plt.subplot(num_rows, num_cols, i + 1)
            plt.imshow((C[k].squeeze()), aspect="auto", vmin=-1, vmax=1)
            plt.axis("off")
            i += 1
        plt.tight_layout()
    else:
        unique_trial_ids = np.unique(np.asarray(trial_ids))
        rep_trial_ids = np.array([])
        for i in range(len(unique_trial_ids)):
            first_idx = np.argmax(trial_ids == unique_trial_ids[i])
            if trial_ids[first_idx] != unique_trial_ids[i]:
                first_idx = -1
                print(f"Warning: trial id {unique_trial_ids[i]} not found in trial_ids array.")
            rep_trial_ids = np.append(rep_trial_ids, first_idx)

        C_averaged = {l: np.zeros_like(C[trial_keys[int(rep_trial_ids[int(l)])]]) for l in unique_trial_ids}

        for k in range(len(trial_ids)):
            C_averaged[int(trial_ids[k])] += np.array(C[trial_keys[k]])

        avg_keys = list(C_averaged.keys())
        num_trials = len(unique_trial_ids)
        num_rows = (num_trials - 1) // 6 + 1
        num_cols = min(6, num_trials)
        plt.figure(figsize=(num_cols, num_rows))
        i=0
        for k in avg_keys:
            plt.subplot(num_rows, num_cols, i + 1)
            plt.imshow((C_averaged[k].squeeze()), aspect="auto", vmin=-1, vmax=1)
            plt.axis("off")
            i += 1
        plt.tight_layout()


def plot_Cs_random_trials(C, num_samples=4, seed=None):
    trial_keys = list(C.keys())
    num_trials = len(trial_keys)
    num_samples = min(num_samples, num_trials)
    rng = np.random.default_rng(seed)
    selected_keys = rng.choice(trial_keys, size=num_samples, replace=False)

    num_cols = min(4, num_samples)
    num_rows = (num_samples - 1) // num_cols + 1
    plt.figure(figsize=(4 * num_cols, 3 * num_rows))

    for i, k in enumerate(selected_keys):
        plt.subplot(num_rows, num_cols, i + 1)
        arr = np.asarray(C[k].squeeze())
        if arr.ndim == 1:
            arr = arr[np.newaxis, :]

        for row_idx in range(arr.shape[0]):
            plt.plot(arr[row_idx, :], label=f"row {row_idx}")

        plt.title(str(k))
        plt.xlabel("column")
        plt.ylabel("c value")
        if arr.shape[0] <= 10:
            plt.legend(fontsize="small", loc="best")

    plt.tight_layout()


def plot_Cs_oneF(C, Fid, trial_ids: np.array = []):
    trial_keys = list(C.keys())
    num_trials = len(C)
    

    unique_trial_ids = np.unique(np.asarray(trial_ids))
    rep_trial_ids = np.array([])
    for i in range(len(unique_trial_ids)):
        first_idx = np.argmax(trial_ids == unique_trial_ids[i])
        if trial_ids[first_idx] != unique_trial_ids[i]:
            first_idx = -1
            print(f"Warning: trial id {unique_trial_ids[i]} not found in trial_ids array.")
        rep_trial_ids = np.append(rep_trial_ids, first_idx)

    C_forF = {l: np.atleast_2d(np.array([])) for l in unique_trial_ids}

    for k in range(len(trial_ids)):
        if C_forF[int(trial_ids[k])].size == 0:
            C_forF[int(trial_ids[k])] = np.atleast_2d(np.array(C[trial_keys[k]][0,Fid,:].squeeze()))
        else:
            C_forF[int(trial_ids[k])] = np.append(np.atleast_2d(C_forF[int(trial_ids[k])]), np.atleast_2d(np.array(C[trial_keys[k]][0,Fid,:].squeeze())),axis=0)

    avg_keys = list(C_forF.keys())
    num_trials = len(unique_trial_ids)
    num_rows = (num_trials - 1) // 6 + 1
    num_cols = min(6, num_trials)
    plt.figure(figsize=(num_cols, num_rows))
    i=0
    for k in avg_keys:
        plt.subplot(num_rows, num_cols, i + 1)
        plt.imshow((C_forF[k].squeeze()), aspect="auto", vmin=-1, vmax=1)
        plt.axis("off")
        i += 1
    plt.tight_layout()


