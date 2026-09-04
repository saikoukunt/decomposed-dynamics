from copyreg import pickle
import os
import pickle
import sys

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
# from plot_utils import plot_Cs_one_F, plot_Fs, plot_Cs, plot_Cs_random_trials
from plot_dlds_outputs import plot_nonzero_slices
import numpy as np
from decomposed_dynamics.dynamics_models import DecomposedLinearDynamics, DecomposedDynamicsModel
from decomposed_dynamics.fitting import fit_no_obs, fit, compute_dynamics_recon_loss
from decomposed_dynamics.inference import bpdn_inference_no_obs
from decomposed_dynamics.inference.base import NoObsInferenceHyperparams
import pickle
from tqdm import tqdm, trange
import random

from datetime import datetime

def infer_no_obs_state_all_trials(
    data: dict,
    dynamics_model: DecomposedDynamicsModel,
    inference_hyperparams: NoObsInferenceHyperparams,
):
    tqdm.write("Final loop to recompute dynamics coeffients")
    trial_keys = list(data.keys())
    C = {}
    pbar = trange(len(trial_keys))

    recon_errs = []
    for i in pbar:
        trial_key = trial_keys[i]
        C[trial_key] = bpdn_inference_no_obs(
            dynamics_model, 
            jnp.expand_dims(data[trial_key], 0), 
            inference_hyperparams
        )
        reconstruction_error = float(
            compute_dynamics_recon_loss(
                dynamics_model, jnp.expand_dims(data[trial_key], 0), C[trial_key]
            )
        )
        recon_errs.append(reconstruction_error)
        pbar.set_postfix(recon_err=f"{reconstruction_error:.4f}")

    return C, np.mean(recon_errs)

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

    prevalenceMap = {}
    for fname in allFiles:
        currAct = np.load(os.path.join(folderName, fname), allow_pickle=True)
        prevalenceMap[fname[:-4]] = 1/currAct.shape[2]
    sumInverseCounts = np.sum(list(prevalenceMap.values()))
    prevalenceMap = {k: v/sumInverseCounts for k, v in prevalenceMap.items()}

    numTrialsAll = []
    allActs = []
    sampling_weights = np.array([])

    for fname in allFiles:
        currAct = np.load(os.path.join(folderName, fname), allow_pickle=True)
        for trial in range(currAct.shape[2]):
            rnnAct[str(kk)] = jnp.array(currAct[:,:,trial].T)
            trial_ids = np.append(trial_ids, ll)
            id_to_task[ll] = fname[:-4]
            sampling_weights = np.append(sampling_weights, prevalenceMap[fname[:-4]])
            kk += 1
        ll += 1
    trial_ids = jnp.array(trial_ids)
    sampling_weights /= np.sum(sampling_weights)

    print(f"size of trial_ids: {trial_ids.shape}")
    print(f"Loaded {len(rnnAct)} trials of RNN activations") # shape (hidden_unit, time)
    
    print("Total trials: ", np.max([int (k) for k in rnnAct.keys()]))
    print("Lenth of input dict: ", len(rnnAct))
    
    # c_l1 = 0.22
    # c_smooth = 0.07

    for c_l1 in [0.1, 0.15, 0.17, 0.2, 0.22, 0.25]:
        for c_smooth in [0.05, 0.07, 0.1]:
            print(f"Computing params: c_l1: {c_l1} | c_smooth: {c_smooth}")
            model = DecomposedLinearDynamics(num_operators=15, num_latents=128, key=jr.key(42))
            inference_hyperparams = NoObsInferenceHyperparams(l1_coeff=c_l1, smooth_coeff=c_smooth)

            with jax.default_device(jax.devices("cuda")[0]):
                model = fit_no_obs(
                    data=rnnAct, 
                    dynamics_model=model,
                    samples_per_snippet=30, 
                    num_snippets=20, 
                    max_iter=1000, 
                    lr_init=1, 
                    inference_hyperparams=inference_hyperparams, 
                    model_update_hyperparams=model.initialize_hyperparams(decorr_coeff=0.03),
                    sampling_weights=sampling_weights
                    # operator_update_hyperparams=model.initialize_hyperparams(decorr_coeff=0.01),
                )
            with jax.default_device(jax.devices("cpu")[0]):
                C_hat, avg_recon_err = infer_no_obs_state_all_trials(
                    dynamics_model=model, 
                    data=rnnAct, 
                    inference_hyperparams=inference_hyperparams
                )

            now = datetime.now().strftime("%Y%m%dT%H%M%S")
            output_dir = f"../outputs/rnn_balancedinputs_newcode"
            os.makedirs(output_dir, exist_ok=True)

            with open('deleteme.pkl', 'wb') as f:  
                pickle.dump([C_hat, trial_ids, id_to_task], f)

            # with open('deleteme.pkl', 'rb') as f: 
            #     C_hat, trial_ids, id_to_task = pickle.load(f)

            # Re-organize data for plotting
            task_coeffs = {v:[] for v in id_to_task.values()}
            trial_ids_np = np.asarray(trial_ids)
            for trialnum, coeffs in C_hat.items():
                taskname = id_to_task[trial_ids_np[int(trialnum)]]
                task_coeffs[taskname].append(coeffs)

            image_outputdir = f"{output_dir}/{now}_cl1{c_l1}_csmooth_{c_smooth}"
            os.makedirs(image_outputdir, exist_ok=True)
            for taskname, coeffs in task_coeffs.items():
                coeffs_np = np.concatenate(coeffs, axis=0).transpose(2, 0, 1)
                task_coeffs[taskname] = coeffs_np                   # (motif, trial, timestep)
                print(f"{taskname}: {coeffs_np.shape}")

                # Plot data
                plot_nonzero_slices(
                    coeffs_np, 
                    title=f"{taskname}\nAvg Reconstruction Err: {avg_recon_err:.2f}", 
                    output_dir=image_outputdir
                )

            with open(f'{output_dir}/rnn_dlds_cl1{c_l1}_csmooth_{c_smooth}_{now}.pkl', 'wb') as f:  
                pickle.dump([model.F, C_hat, trial_ids, c_l1, id_to_task], f)