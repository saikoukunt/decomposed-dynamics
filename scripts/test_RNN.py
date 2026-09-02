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

    numTrialsAll = []
    allActs = []
    for ll, fname in enumerate(allFiles):
        currAct = np.load(os.path.join(folderName, fname), allow_pickle=True)
        id_to_task[ll] = fname[:-4]

        allActs.append(currAct)
        numTrialsAll.append(currAct.shape[2])


    # ### Balance the inputs so that the number of trials per task
    # ### is equal to the minimum number of trials in a single task
    # numTrials = np.min(numTrialsAll)
    # print(f"Loading {numTrials} trials per task")
    # for ll, currAct in enumerate(allActs):
    #     if currAct.shape[2] > numTrials:
    #         trial_ids_toadd = random.sample(range(currAct.shape[2]), numTrials)
    #     else:
    #         trial_ids_toadd = range(currAct.shape[2])
        
    #     for trial in trial_ids_toadd:
    #         rnnAct[str(kk)] = jnp.array(currAct[:,:,trial])
    #         trial_ids = np.append(trial_ids, ll)
    #         kk += 1

    numTrials = np.max(numTrialsAll)

    for ll, currAct in enumerate(allActs):
        n_trials = currAct.shape[2]

        trial_ids_toadd = np.random.choice(
            n_trials,
            size=numTrials,
            replace=(n_trials < numTrials),
        )

        for trial in trial_ids_toadd:
            rnnAct[str(kk)] = jnp.array(currAct[:, :, trial])
            trial_ids = np.append(trial_ids, ll)
            kk += 1
            
    trial_ids = jnp.array(trial_ids)
    print(f"size of trial_ids: {trial_ids.shape}")
    print(f"Loaded {len(rnnAct)} trials of RNN activations, each with shape {rnnAct["0"].shape}.") # shape (hidden_unit, time)
    
    c_l1 = 0.3
    c_smooth = 0.05

    # for c_l1 in [0.1, 0.15, 0.2, 0.25, 0.3, 0.4]:
    #     for c_smooth in [0.05, 0.1, 0.15, 0.2]:
    # for _ in range(5):
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
            F_decorr_coeff=0.012     # stop it from learning the same thing in 2 motifs
        )
    with jax.default_device(jax.devices("cpu")[0]):
        C_hat = infer_no_obs_state_all_trials(rnnAct, F_hat, c_l1_coeff=c_l1, c_smooth_coeff=c_smooth)

    now = datetime.now().strftime("%Y%m%dT%H%M%S")
    output_dir = f"../outputs/rnn_balancedinputs"
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
