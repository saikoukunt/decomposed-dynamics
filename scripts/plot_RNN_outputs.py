from copyreg import pickle
import os
import pickle
import sys

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from plot_utils import plot_Cs_one_F, plot_Fs, plot_Cs, plot_Cs_random_trials, repackage_C_hat
import numpy as np
from decomposed_dynamics.dlds import fit_no_obs, infer_no_obs_state_all_trials
import pickle


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    folderName = "/home/adam/Dropbox/Multi-Task-RNN/rnn_hidden_states"
    allFiles = sorted(f for f in os.listdir(folderName) if f.endswith(".npy"))
    #rnnAct     = {}
    #trial_ids  = np.array([])
    #kk = 0
    #ll = 0
    #for fname in allFiles:
    #    currAct = np.load(os.path.join(folderName, fname), allow_pickle=True)
    #    for trial in range(currAct.shape[2]):
    #        rnnAct[str(kk)] = jnp.array(currAct[:,:,trial])
    #        trial_ids = np.append(trial_ids, ll)
    #        kk += 1
    #    ll += 1
    #trial_ids = jnp.array(trial_ids)

    with open('dLDS_output_jax_2.pkl', 'rb') as f: 
        F_hat, C_hat, trial_ids, c_l1, c_smooth = pickle.load(f)
            # repackage into trial-based aggregation of C's for saving
    C_hat_repackaged = repackage_C_hat(C_hat, trial_ids)

    print(f"Loaded {len(C_hat_repackaged)} trials of inferred dynamics coefficients, each with shape {C_hat_repackaged[0].shape}.")
    print(f"Plotting results...")
    #plot_Cs_oneF(C_hat, trial_ids=trial_ids, Fid=2)
    #plot_Cs_random_trials(C_hat, 10)
    #plot_Fs(F_hat)
    #plt.show()



