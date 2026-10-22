from copyreg import pickle
import os
import pickle
import sys

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
# from plot_utils import plot_Cs_one_F, plot_Fs, plot_Cs, plot_Cs_random_trials
from plot_dlds_outputs import plot_nonzero_slices
import numpy as np
from decomposed_dynamics.dynamics_models import DecomposedLinearDynamics, DecomposedDynamicsModel
from decomposed_dynamics.fitting import fit_no_obs, fit, compute_dynamics_recon_loss
from decomposed_dynamics.inference import bpdn_inference_no_obs #bpdn_inference_no_obs
from decomposed_dynamics.inference.base import InferenceHyperparams #NoObsInferenceHyperparams
import pickle
from tqdm import tqdm, trange
import random

from datetime import datetime

def infer_no_obs_state_all_trials(
    data: dict,
    dynamics_model: DecomposedDynamicsModel,
    inference_hyperparams: InferenceHyperparams,
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

    folderName = "/cis/home/jye/Workspace/dLDS-multitask-rnn/dLDS/rnn_hidden_states"
    allFiles   = sorted(f for f in os.listdir(folderName) if f.endswith(".npy"))
    # allFiles   = ["delaygo.npy", "delayanti.npy", "fdgo.npy", "fdanti.npy"]
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
    
    data_descriptions = {}
    for fname in allFiles:
        currAct = np.load(os.path.join(folderName, fname), allow_pickle=True)
        print(fname[:-4], currAct.shape)
        for trial in range(currAct.shape[2]):
            rnnAct[str(kk)] = jnp.array(currAct[:,:,trial].T) # (time, hidden_unit)
            trial_ids = np.append(trial_ids, ll)
            id_to_task[ll] = fname[:-4]
            sampling_weights = np.append(sampling_weights, prevalenceMap[fname[:-4]])
            if fname[:-4] not in data_descriptions.keys():
                data_descriptions[fname[:-4]]= currAct.shape
            kk += 1
        ll += 1
    trial_ids = jnp.array(trial_ids)
    sampling_weights /= np.sum(sampling_weights)

    # NORMALIZE
    # Concatenate all trials along the time axis
    all_acts = jnp.concatenate(list(rnnAct.values()), axis=0)

    # Compute statistics per hidden unit
    mean = jnp.mean(all_acts, axis=0, keepdims=True)
    std = jnp.std(all_acts, axis=0, keepdims=True)

    # Normalize each trial
    rnnAct = {
        k: (v - mean) / jnp.maximum(std, 1e-8)
        for k, v in rnnAct.items()
    }

    print(f"size of trial_ids: {trial_ids.shape}")
    
    # print("Data sizes: ", data_descriptions)
    
    # c_l1 = 0.22
    # c_smooth = 0.07

    # for c_l1 in [0.1, 0.15, 0.17, 0.2, 0.22, 0.25]:
    #     for c_smooth in [0.05,0.1,0.15,0.2,0.3]:

    # L1 regularization needs to modify itself 
    # Could also normalize data
    # Smaller values admit more motifs, not penalizing C's as much
    # Find the transition point between 2 motifs and a ton
    # Find a spot where we see a lot of motifs, then play around with the model hyperparms (deocrr, l1_reweight...)
    # Does decorrelation coeff also cover self correlation? (can help remove extraneous dynamics)
    dlc = 1
    l1_rw_coeff = 100
    # c_l1 = 0.27
    decorr_coeff = 0.01
    for l1_coeff in [1e-3, 1e-5]:#[0.0, 0.01, 0.02, 0.05]: #small 1e-5
        for c_smooth in [1e-3, 1e-5]:#[0.058, 0.059, 0.06, 0.061, 0.062]: #small 1e-5
            # for c_l1 in [0.01, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6]:
            for c_l1 in [0.5, 0.7, 0.9, 1, 1.5, 2, 2.5, 3, 5]:
                print(f"Computing params: c_l1: {c_l1} | c_smooth: {c_smooth}")
                model = DecomposedLinearDynamics(num_operators=15, num_latents=128, key=jr.key(42))
                inference_hyperparams = InferenceHyperparams(
                    l1_coeff=c_l1, 
                    smooth_coeff=c_smooth,
                    dynamics_loss_coeff=dlc,
                    solver="proximal",
                    # spgl1_sigma=0.1,
                    # spgl1_max_iter=30,
                    # spgl1_outer_iter=8,
                )

                with jax.default_device(jax.devices("cuda")[0]):
                    model = fit_no_obs(
                        data=rnnAct, 
                        dynamics_model=model,
                        samples_per_snippet=80, #30, 
                        num_snippets=15, 
                        max_iter=300, 
                        lr_init=1, 
                        inference_hyperparams=inference_hyperparams, 
                        model_update_hyperparams=model.initialize_hyperparams(
                                                            decorr_coeff=decorr_coeff, # Modulate the number of dyanmical systems
                                                            l1_coeff=l1_coeff, 
                                                            l1_reweight_coeff=l1_rw_coeff
                                                        ), 
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
                output_dir = f"../outputs/rnn_balancedinputs_newcode_v16_proxnnlasso_zscore"
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

                image_outputdir = f"{output_dir}/{now}_cl1{c_l1}_csmooth_{c_smooth}_dlc{dlc}_lwrw{l1_rw_coeff}_decorr{decorr_coeff}_fl1{l1_coeff}"
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

                with open(f'{output_dir}/rnn_dlds_cl1{c_l1}_csmooth_{c_smooth}_dlc{dlc}_lwrw{l1_rw_coeff}_decorr{decorr_coeff}_fl1{l1_coeff}_{now}.pkl', 'wb') as f:  
                    pickle.dump([model.F, C_hat, trial_ids, c_l1, id_to_task], f)