from copyreg import pickle
import os
import pickle
import sys

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
# from plot_utils import plot_Cs_one_F, plot_Fs, plot_Cs, plot_Cs_random_trials
from plot_dlds_outputs import plot_nonzero_slices
import numpy as np
from decomposed_dynamics.dlds import fit_no_obs, infer_no_obs_state_all_trials
import pickle

from datetime import datetime

if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    output_dir = f"../outputs/belief_transformer_trainlen25_v3_norm75"
    os.makedirs(output_dir, exist_ok=True)

    folderName = "/home/yejz1/workspace/jhu/transformer_belief/outputs/train_seqlen25_v2/activations"
    allFiles = sorted(f for f in os.listdir(folderName) if f.endswith(".npy"))
    print(allFiles)
    transformerAct     = {}
    trial_ids  = np.array([])
    id_to_task = {}
    kk = 0
    ll = 0
    for fname in allFiles:
        print(f"Loading {os.path.join(folderName, fname)}")
        currAct = np.load(os.path.join(folderName, fname), allow_pickle=True)
        print(f"Data shape: {currAct.shape}")
        id_to_task[ll] = fname[:-4]

        # Activations are of dim (trial, squence/time, hidden_units)
        for trial in range(currAct.shape[0]):
            # transformerAct[str(kk)] = jnp.array(currAct[trial,:,:].T)

            # Normalize activations so that values in the 75th percentile
            # are normalized to 1
            arr = jnp.array(currAct[trial, :, :].T)
            p75 = jnp.percentile(jnp.abs(arr), 75)
            transformerAct[str(kk)] = arr / jnp.maximum(p75, 1e-8)

            trial_ids = np.append(trial_ids, ll)
            kk += 1
        print(f"Finished loading trials of shape: {transformerAct[str(kk-1)].shape}")
        ll += 1
    trial_ids = jnp.array(trial_ids)

    print(f"size of trial_ids: {trial_ids.shape}")
    print(f"{fname}: Loaded {len(transformerAct)} trials of transformer activations.") 
    # Expected activations shape is (hidden_units, time/seqlen)

    c_l1 = 0.2
    c_smooth = 0.01

    # for c_l1 in [0.1, 0.15, 0.2, 0.25, 0.3, 0.4]:
    #     for c_smooth in [0.05, 0.1, 0.15, 0.2]:
    # for _ in range(5):
    with jax.default_device(jax.devices("cuda")[0]):
        F_hat = fit_no_obs(
            data=transformerAct, 
            num_motifs=10, 
            samples_per_snippet=30, 
            num_snippets=150, 
            max_iter=1000, 
            F_lr_init=.1, 
            c_l1_coeff=c_l1, 
            c_smooth_coeff=c_smooth, 
            F_l1_coeff=0.001,
            F_decorr_coeff=0.0
        )
    with jax.default_device(jax.devices("cpu")[0]):
        C_hat = infer_no_obs_state_all_trials(transformerAct, F_hat, c_l1_coeff=c_l1, c_smooth_coeff=c_smooth)

    now = datetime.now().strftime("%Y%m%dT%H%M%S")

    # Re-organize data for plotting
    task_coeffs = {v:[] for v in id_to_task.values()}
    trial_ids_np = np.asarray(trial_ids)
    for trialnum, coeffs in C_hat.items():
        taskname = id_to_task[trial_ids_np[int(trialnum)]]
        task_coeffs[taskname].append(coeffs)

    image_outputdir = f"{output_dir}/{now}_cl1{c_l1}_csmooth{c_smooth}"
    os.makedirs(image_outputdir, exist_ok=True)
    for taskname, coeffs in task_coeffs.items():
        coeffs_np = np.concatenate(coeffs, axis=0).transpose(1, 0, 2)
        task_coeffs[taskname] = coeffs_np                   # (motif, trial, timestep)
        print(f"Plotting {taskname}: {coeffs_np.shape}")

        # Plot data
        plot_nonzero_slices(coeffs_np, title=taskname, output_dir=image_outputdir)

    with open(f'{output_dir}/rnn_dlds_cl1{c_l1}_csmooth{c_smooth}_{now}.pkl', 'wb') as f:  
        pickle.dump([F_hat, C_hat, trial_ids, c_l1, c_smooth, id_to_task], f)
