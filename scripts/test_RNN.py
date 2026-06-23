from copyreg import pickle
import os
import pickle
import sys

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from plot_utils import plot_Cs_oneF, plot_Fs, plot_Cs, plot_Cs_random_trials
import numpy as np
from decomposed_dynamics.dlds import fit_no_obs, final_c_fit
import pickle

if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    folderName = "/home/adam/Dropbox/Multi-Task-RNN/rnn_hidden_states"
    allFiles = sorted(f for f in os.listdir(folderName) if f.endswith(".npy"))
    rnnAct     = {}
    trial_ids  = np.array([])
    kk = 0
    ll = 0
    for fname in allFiles:
        currAct = np.load(os.path.join(folderName, fname), allow_pickle=True)
        for trial in range(currAct.shape[2]):
            rnnAct[str(kk)] = jnp.array(currAct[:,:,trial])
            trial_ids = np.append(trial_ids, ll)
            kk += 1
        ll += 1
    trial_ids = jnp.array(trial_ids)

    print(f"size of trial_ids: {trial_ids.shape}")
    print(f"Loaded {len(rnnAct)} trials of RNN activations, each with shape {rnnAct["0"].shape}.")
    
    c_l1 = 0.3;
    c_smooth = 0.1;
    with jax.default_device(jax.devices("cpu")[0]):
        F_hat = fit_no_obs(rnnAct, 20, 10, 10, max_iter=10, F_lr_init=10, c_l1_coeff=c_l1, c_smooth_coeff=c_smooth, F_l1_coeff=0.001,F_decorr_coeff=0.0)
        C_hat = final_c_fit(rnnAct, F_hat, c_l1_coeff=c_l1, c_smooth_coeff=c_smooth)

        print(f"run finished, plotting results...")
        plot_Cs_oneF(C_hat, trial_ids=trial_ids, Fid=2)
        plot_Cs_random_trials(C_hat, 10)
        plot_Fs(F_hat)
        plt.show()

    # repackage into trial-based aggregation of C's for saving
    C_hat_repackaged = {}
    for i, trial_id in enumerate(trial_ids):
        C_hat_repackaged[str(int(trial_id))] = np.append(C_hat[i])
    # Save the data ouput to a pickle file
    with open('dLDS_output_jax_2.pkl', 'wb') as f:  
        pickle.dump([F_hat, C_hat, trial_ids, c_l1, c_smooth], f)
