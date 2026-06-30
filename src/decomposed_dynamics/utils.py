def repackage_C_hat(C_hat, trial_ids):
    """
    Repackage the inferred dynamics coefficients C_hat into a dictionary where each key corresponds to a unique trial_id.
    
    Parameters:
    - C_hat: dict
        A dictionary containing inferred dynamics coefficients for each trial.
    - trial_ids: array-like
        An array of trial IDs corresponding to the trials in C_hat.
    
    Returns:
    - C_hat_repackaged: dict
        A dictionary where each key is a unique trial_id and the value is an array of dynamics coefficients for that trial.
    """
    C_hat_repackaged = {i: np.array([]) for i in np.unique(trial_ids)}
    print(f"Repackaging {len(C_hat)} inferred dynamics coefficients into {len(C_hat_repackaged)} = {np.unique(trial_ids)} trials...")
    Ckeys = list(C_hat.keys())
    for i, trial_id in enumerate(trial_ids):
        Ctmp = np.atleast_3d(C_hat[Ckeys[i]].squeeze())
        print(f"trial_id: {trial_id}, Ctmp.shape: {Ctmp.shape}")
        print(f"trial_id: {trial_id}, C_hat_repackaged[{int(trial_id)}].shape: {np.atleast_3d(C_hat_repackaged[int(trial_id)]).shape}")
        if len(C_hat_repackaged[int(trial_id)]) == 0:
            C_hat_repackaged[int(trial_id)] = np.atleast_3d(np.array(C_hat[Ckeys[i]].squeeze()))
        else:
            C_hat_repackaged[int(trial_id)] = np.append(np.atleast_3d(C_hat_repackaged[int(trial_id)]),
                                                    np.atleast_3d(np.array(C_hat[Ckeys[i]].squeeze())),axis=2)

    return C_hat_repackaged
