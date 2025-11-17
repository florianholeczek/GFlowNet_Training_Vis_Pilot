import numpy as np
import pandas as pd
import random

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdmolops
from rdkit import DataStructs
import random
import os
from tqdm import tqdm
from rdkit.Chem import rdFingerprintGenerator
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

def generate_data_OLD(
        n_unique=5000,
        repeat_fraction=0.3,
        iterations=np.arange(500, 10001, 500),
        seed=42,
        save_csv=False
):
    """
    Generate mock GFlowNet-like molecule and trajectory datasets.
    Trajectories reflect the incremental construction of the final SMILES.

    Returns
    -------
    molecules_df : pd.DataFrame
        Columns: Object, Reward1, Reward2, Reward3, RewardTotal, Iteration
    trajectories_df : pd.DataFrame
        Columns: Object, Step, State
    """
    np.random.seed(seed)
    random.seed(seed)

    # Define token sets
    atoms = ['C', 'N', 'O', 'F', 'Cl', 'Br', 'S']
    bonds = ['', '=', '#']
    rings = list(map(str, range(1, 10)))

    # Storage for molecules + trajectories
    smiles_list = []
    traj_records = []

    # --- Generate molecules and trajectories simultaneously ---
    for _ in range(n_unique):
        length = np.random.randint(5, 25)
        s = ''
        steps = []
        for _ in range(length):
            token_type = np.random.choice(['atom', 'bond', 'ring'], p=[0.7, 0.2, 0.1])
            if token_type == 'atom':
                token = random.choice(atoms)
            elif token_type == 'bond':
                token = random.choice(bonds)
            else:
                token = random.choice(rings)
            s += token
            steps.append(s)

        smiles_list.append(s)
        for step, state in enumerate(steps):
            traj_records.append({"Object": s, "Step": step, "State": state})

    trajectories_df = pd.DataFrame(traj_records)
    # FLows
    trajectories_df['flows_forward'] = np.random.exponential(scale=2, size= len(trajectories_df)).clip(0.01, 10)
    trajectories_df['flows_backward'] = np.random.exponential(scale=2, size=len(trajectories_df)).clip(0.01, 10)

    #states, ids, final flows
    trajectories_df["state"] = np.where(trajectories_df["Step"] == 0, "start", "intermediate")
    max_steps = trajectories_df.groupby("Object")["Step"].transform("max")
    trajectories_df.loc[trajectories_df["Step"] == max_steps, "state"] = "end"
    trajectories_df["trajectory_id"] = (trajectories_df["Object"] != trajectories_df["Object"].shift()).cumsum()
    trajectories_df.loc[trajectories_df["state"] == "end", ["flows_forward", "flows_backward"]] = np.nan


    # --- Rewards (correlated/uncorrelated) ---
    n_unique = len(smiles_list)
    r1 = np.random.rand(n_unique)
    r2 = 0.6 * r1 + 0.4 * np.random.rand(n_unique)
    r3 = np.random.rand(n_unique)
    r_total = (r1 + r2 + r3) / 3

    molecules_df = pd.DataFrame({
        "Object": smiles_list,
        "Reward1": r1,
        "Reward2": r2,
        "Reward3": r3,
        "RewardTotal": r_total
    })

    # --- Weighted sampling for repeats ---
    n_total = int(n_unique * (1 + repeat_fraction))
    weights = molecules_df["RewardTotal"] / molecules_df["RewardTotal"].sum()
    sampled = molecules_df.sample(n=n_total, replace=True, weights=weights, random_state=seed)

    # Make Iteration depend on reward
    sampled_iterations = []
    for r in sampled["RewardTotal"]:
        # Higher reward -> more likely higher iterations
        iteration_weights = np.linspace(1, 10, len(iterations)) * r
        iteration_prob = iteration_weights / iteration_weights.sum()
        sampled_iterations.append(np.random.choice(iterations, p=iteration_prob))
    sampled["Iteration"] = sampled_iterations
    sampled = sampled.sample(frac=1, random_state=seed + 1).reset_index(drop=True)

    # --- Optionally save CSVs ---
    if save_csv:
        sampled.to_csv("data_objects.csv", index=False)
        trajectories_df.to_csv("data_trajectories.csv", index=False)
        print("Saved data as csv")

    return sampled, trajectories_df




def create_molecule_dataset(
    input_csv,
    n_unique=100,
    repeat_fraction=0.2,
    output_dir='.',
    fingerprint_bits=1024,
    fingerprint_radius=2,
    random_seed=42
):
    """
    Create a dataset simulating sequential generation of molecules.

    Parameters
    ----------
    input_csv : str
        Path to input CSV with a 'smiles' column.
    n_unique : int
        Number of unique molecules to sample.
    repeat_fraction : float
        Fraction of repeats to add (duplicates keep the same rewards).
    output_dir : str
        Directory where output CSVs will be saved.
    fingerprint_bits : int
        Length of the Morgan fingerprint.
    fingerprint_radius : int
        Radius for Morgan fingerprint.
    random_seed : int
        Random seed for reproducibility.
    """

    np.random.seed(random_seed)
    random.seed(random_seed)
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=fingerprint_radius, fpSize=fingerprint_bits)

    # Detect whether the input is a CSV or plain text file
    if input_csv.endswith(".csv"):
        df = pd.read_csv(input_csv)
        if 'smiles' not in df.columns:
            raise ValueError("Input CSV must have a column named 'smiles'.")
        smiles_list = df['smiles'].astype(str).tolist()
    else:
        # Plain text file: one SMILES per line
        with open(input_csv, 'r') as f:
            smiles_list = [line.strip() for line in f if line.strip()]

    smiles_list = smiles_list[:n_unique+100]

    # --- Step 1: Select valid molecules ---
    print("1. Selecting")
    mols = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is not None:
            mols.append((smi, mol))
    if len(mols) < n_unique:
        raise ValueError(f"Not enough valid molecules ({len(mols)}) for n_unique={n_unique}.")

    mols = random.sample(mols, n_unique)

    smiles = [m[0] for m in mols]
    mol_objs = [m[1] for m in mols]
    data = pd.DataFrame({'smiles': smiles})

    # --- Step 2: Generate correlated rewards ---
    print("2. rewards")
    mean = [0.5, 0.5]
    cov = [[0.02, 0.01],
           [0.01, 0.02]]  # moderate correlation (r ~ 0.5)
    r1_r2 = np.random.multivariate_normal(mean, cov, size=n_unique)
    r3 = np.random.normal(0.5, 0.15, size=n_unique)
    reward1, reward2 = np.clip(r1_r2[:, 0], 0, 1), np.clip(r1_r2[:, 1], 0, 1)
    reward3 = np.clip(r3, 0, 1)
    data['reward1'] = reward1
    data['reward2'] = reward2
    data['reward3'] = reward3
    data['reward_total'] = (reward1 + reward2 + reward3) / 3

    # --- Step 3: Duplicate some molecules ---
    print("3. duplicating")
    n_repeat = int(n_unique * repeat_fraction)
    if n_repeat > 0:
        dup_indices = np.random.choice(n_unique, n_repeat, replace=True)
        duplicates = data.iloc[dup_indices].copy()
        data = pd.concat([data, duplicates], ignore_index=True)

    # Assign unique IDs
    data['id'] = range(1, len(data) + 1)
    data['iteration'] = data['id']%20 * 500

    # --- Step 4: Generate fingerprints ---
    print("4. fingerprints")
    fps = []
    for smi in data['smiles']:
        mol = Chem.MolFromSmiles(smi)
        fp = fpgen.GetFingerprint(mol)
        arr = np.zeros((fingerprint_bits,), dtype=int)
        DataStructs.ConvertToNumpyArray(fp, arr)
        fps.append(arr)
    fp_df = pd.DataFrame(fps, columns=[f'fp{i}' for i in range(fingerprint_bits)])
    data_objects = pd.concat([data, fp_df], axis=1)

    # --- Step 5: Deconstruct molecules to build trajectories ---
    print("5. deconstruct")
    traj_records = []

    for _, row in tqdm(data_objects.iterrows()):
        smi = row['smiles']
        mol = Chem.MolFromSmiles(smi)
        final_id = row['id']
        step = 0

        # final molecule
        traj_records.append({
            'final_id': final_id,
            'step': step,
            'smiles': smi,
            'iteration': row['iteration'],
            'state': 'final',
            'valid': True
        })

        current_mol = Chem.Mol(mol)

        while current_mol.GetNumAtoms() > 1:
            # Step 1: find leaves
            leaves = [a.GetIdx() for a in current_mol.GetAtoms() if a.GetDegree() == 1]

            if leaves:
                # leaf-first: remove smallest-index leaf
                atom_to_remove = min(leaves)
            else:
                # no leaves left: remove smallest-index internal atom
                atom_to_remove = 0  # canonical order

            emol = Chem.EditableMol(current_mol)
            emol.RemoveAtom(atom_to_remove)
            new_mol = emol.GetMol()

            # optional: check if chemically valid
            try:
                Chem.SanitizeMol(new_mol)
                valid = True
            except:
                valid = False

            step += 1
            new_smi = Chem.MolToSmiles(new_mol) if new_mol.GetNumAtoms() > 0 else ''
            state = 'intermediate' if new_mol.GetNumAtoms() > 1 else 'start'

            traj_records.append({
                'final_id': final_id,
                'step': step,
                'smiles': new_smi,
                'iteration': row['iteration'],
                'state': state,
                'valid': valid
            })

            # update for next iteration
            current_mol = new_mol

            if current_mol.GetNumAtoms() == 0:
                break  # molecule fully deconstructed

    data_trajectories = pd.DataFrame(traj_records)
    # Flows
    data_trajectories['flows_forward'] = np.random.exponential(scale=2, size=len(data_trajectories)).clip(0.01, 10)
    data_trajectories['flows_backward'] = np.random.exponential(scale=2, size=len(data_trajectories)).clip(0.01, 10)
    #data_trajectories.loc[data_trajectories["state"] == "final", ["flows_forward", "flows_backward"]] = np.nan

    # --- Step 6: Fingerprints for trajectory molecules ---
    print("6. trajectories fingerprints")
    fps_traj = []
    for smi in tqdm(data_trajectories['smiles']):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            arr = np.zeros((fingerprint_bits,), dtype=int)
        else:
            fp = fpgen.GetFingerprint(mol)
            arr = np.zeros((fingerprint_bits,), dtype=int)
            DataStructs.ConvertToNumpyArray(fp, arr)
        fps_traj.append(arr)
    fp_traj_df = pd.DataFrame(fps_traj, columns=[f'fp{i}' for i in range(fingerprint_bits)])
    data_trajectories = pd.concat([data_trajectories, fp_traj_df], axis=1)

    # --- Step 7: Save results ---
    print("7. saving")
    os.makedirs(output_dir, exist_ok=True)
    data_objects.to_csv(os.path.join(output_dir, 'data_objects.csv'), index=False)
    data_trajectories.to_csv(os.path.join(output_dir, 'data_trajectories.csv'), index=False)

    print(f"Saved {len(data_objects)} objects and {len(data_trajectories)} trajectory steps to {output_dir}.")

    return data_objects, data_trajectories





if __name__ == "__main__":
    """
    molecules, trajectories = generate_data(
            n_unique=7000,
            repeat_fraction=0.3,
            iterations=np.arange(500, 10001, 500),
            seed=42,
            save_csv=True
    )
    """
    create_molecule_dataset(
        "smiles_start.txt",
        n_unique=100,
        repeat_fraction=1.2,
        output_dir='.',
        fingerprint_bits=1024,
        fingerprint_radius=2,
        random_seed=42
    )
