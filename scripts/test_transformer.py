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
import pickle

from datetime import datetime

from decomposed_dynamics.dynamics_models import DecomposedLinearDynamics, DecomposedDynamicsModel
from decomposed_dynamics.fitting import fit_no_obs, fit, compute_dynamics_recon_loss
from decomposed_dynamics.inference import bpdn_inference_no_obs
from decomposed_dynamics.inference.base import NoObsInferenceHyperparams

from test_RNN import infer_no_obs_state_all_trials

if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    output_dir = f"../outputs/belief_transformer_switching"
    os.makedirs(output_dir, exist_ok=True)

    # folderName = "/home/yejz1/workspace/jhu/transformer_belief/outputs/train_seqlen25/activations"
    folderName = "/cis/home/jye/Workspace/belief-transformer/outputs/transformer_switching/run_seqlen10_20260917113621"
    allFiles = sorted(f for f in os.listdir(folderName) if f.startswith("activations") and f.endswith(".npy"))
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
            transformerAct[str(kk)] = jnp.array(currAct[trial,:,:])
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

    model = DecomposedLinearDynamics(num_operators=15, num_latents=64, key=jr.key(42))
    inference_hyperparams = NoObsInferenceHyperparams(l1_coeff=c_l1, smooth_coeff=c_smooth)

    with jax.default_device(jax.devices("cuda")[0]):
        model = fit_no_obs(
            data=transformerAct, 
            dynamics_model=model,
            samples_per_snippet=30, 
            num_snippets=20, 
            max_iter=1000, 
            lr_init=1, 
            inference_hyperparams=inference_hyperparams, 
            model_update_hyperparams=model.initialize_hyperparams(decorr_coeff=0.01, l1_coeff=0.0),
            # sampling_weights=sampling_weights
            # operator_update_hyperparams=model.initialize_hyperparams(decorr_coeff=0.01),
        )
    with jax.default_device(jax.devices("cpu")[0]):
        C_hat, avg_recon_err = infer_no_obs_state_all_trials(
            dynamics_model=model, 
            data=transformerAct, 
            inference_hyperparams=inference_hyperparams
        )

    now = datetime.now().strftime("%Y%m%dT%H%M%S")
    output_dir = f"../outputs/transformer_switching"
    os.makedirs(output_dir, exist_ok=True)

    with open('deleteme_transformer.pkl', 'wb') as f:  
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

    with open(f'{output_dir}/transformer_dlds_cl1{c_l1}_csmooth_{c_smooth}_{now}.pkl', 'wb') as f:  
        pickle.dump([model.F, C_hat, trial_ids, c_l1, id_to_task], f)
