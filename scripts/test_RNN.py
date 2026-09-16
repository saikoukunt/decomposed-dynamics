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
import random
from collections import Counter
from datetime import datetime

if __name__ == "__main__":
    # seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    random.seed(42)

    folderName = "/home/yejz1/workspace/jhu/dlds/rnn_hidden_states"
    allFiles = sorted(f for f in os.listdir(folderName) if f.endswith(".npy"))
    rnnAct     = {}
    trial_ids  = np.array([])
    id_to_task = {}
    kk = 0
    ll = 0

    for fname in allFiles:
        currAct = np.load(os.path.join(folderName, fname), allow_pickle=True)
        for trial in range(currAct.shape[2]):
            rnnAct[str(kk)] = jnp.array(currAct[:,:,trial])
            trial_ids = np.append(trial_ids, ll)
            if ll not in id_to_task.keys():
                id_to_task[ll] = fname[:-4]
            kk += 1
        ll += 1

    trial_ids = np.asarray(trial_ids)
    trial_counts = Counter(trial_ids)
    trial_weights = np.array(
        [1.0 / trial_counts[task] for task in trial_ids],
        dtype=np.float64,
    )
    trial_probs = trial_weights / trial_weights.sum()
    # trial_probs = np.ones(len(trial_ids)) / len(trial_ids)

    trial_ids = jnp.array(trial_ids)

    print(f"size of trial_ids: {trial_ids.shape}")

    c_l1 = 0.3
    c_smooth = 0.07

    # for c_l1 in [0.1, 0.15, 0.2, 0.25, 0.3, 0.4]:
    #     for c_smooth in [0.05, 0.1, 0.15, 0.2]:
    for c_l1 in [0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]:
        with jax.default_device(jax.devices("cuda")[0]):
            F_hat = fit_no_obs(
                data=rnnAct, 
                num_motifs=20, 
                samples_per_snippet=30, 
                num_snippets=20, 
                max_iter=1000, 
                F_lr_init=1, 
                c_l1_coeff=c_l1, 
                c_smooth_coeff=c_smooth, 
                F_l1_coeff=0.000,       # sparsity on F's (latent space only)
                F_decorr_coeff=0.01,     # stop it from learning the same thing in 2 motifs
                trial_probs=trial_probs
            )
        with jax.default_device(jax.devices("cpu")[0]):
            C_hat = infer_no_obs_state_all_trials(rnnAct, F_hat, c_l1_coeff=c_l1, c_smooth_coeff=c_smooth)

        now = datetime.now().strftime("%Y%m%dT%H%M%S")
        output_dir = f"../outputs/rnn_weightedinputs"
        os.makedirs(output_dir, exist_ok=True)

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
            print(f"{taskname}: {coeffs_np.shape}")

            # Plot data
            plot_nonzero_slices(coeffs_np, title=taskname, output_dir=image_outputdir)

        with open(f'{output_dir}/rnn_dlds_cl1{c_l1}_csmooth{c_smooth}_{now}.pkl', 'wb') as f:  
            pickle.dump([F_hat, C_hat, trial_ids, c_l1, c_smooth, id_to_task], f)
