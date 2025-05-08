import streamlit as st
import pandas as pd
from multiprocessing import Pool, cpu_count
from rdkit import Chem
from rdkit.Chem import AllChem
from pyscf import gto, dft
from pyscf.hessian import thermo
from pyscf.geomopt import geometric_solver
from datetime import datetime
import os
import numpy as np

# --- Constants ---
METHOD = 'B3LYP'
BASIS = 'STO-3G'
TEMPERATURES = [293.15, 323.15]
DIELECTRIC_MAP = {
    "water": 78.3553,
    "acetonitrile": 35.688,
    "methanol": 32.613,
    "ethanol": 24.852,
    "isoquinoline": 11.0,
    "quinoline": 9.16,
    "chloroform": 4.7113,
    "diethylether": 4.24,
    "dichloromethane": 8.93,
    "dichloroethane": 10.125,
    "carbontetrachloride": 2.228,
    "benzene": 2.2706,
    "toluene": 2.3741,
    "chlorobenzene": 5.6968,
    "nitromethane": 36.562,
    "heptane": 1.9113,
    "cyclohexane": 2.0165,
    "aniline": 6.8882,
    "acetone": 20.493,
    "tetrahydrofuran": 7.4257,
    "dimethylsulfoxide": 46.826,
    "argon": 1.43,
    "krypton": 1.519,
    "xenon": 1.706,
    "n-octanol": 9.8629,
    "1,1,1-trichloroethane": 7.0826,
    "1,1,2-trichloroethane": 7.1937,
    "1,2,4-trimethylbenzene": 2.3653,
    "1,2-dibromoethane": 4.9313,
    "1,2-ethanediol": 40.245,
    "1,4-dioxane": 2.2099,
    "1-bromo-2-methylpropane": 7.7792,
    "1-bromooctane": 5.0244,
    "1-bromopentane": 6.269,
    "1-bromopropane": 8.0496,
    "1-butanol": 17.332,
    "1-chlorohexane": 5.9491,
    "1-chloropentane": 6.5022,
    "1-chloropropane": 8.3548,
    "1-decanol": 7.5305,
    "1-fluorooctane": 3.89,
    "1-heptanol": 11.321,
    "1-hexanol": 12.51,
    "1-hexene": 2.0717,
    "1-hexyne": 2.615,
    "1-iodobutane": 6.173,
    "1-iodohexadecane": 3.5338,
    "1-iodopentane": 5.6973,
    "1-iodopropane": 6.9626,
    "1-nitropropane": 23.73,
    "1-nonanol": 8.5991,
    "1-pentanol": 15.13,
    "1-pentene": 1.9905,
    "1-propanol": 20.524,
    "2,2,2-trifluoroethanol": 26.726,
    "2,2,4-trimethylpentane": 1.9358,
    "2,4-dimethylpentane": 1.8939,
    "2,4-dimethylpyridine": 9.4176,
    "2,6-dimethylpyridine": 7.1735,
    "2-bromopropane": 9.361,
    "2-butanol": 15.944,
    "2-chlorobutane": 8.393,
    "2-heptanone": 11.658,
    "2-hexanone": 14.136,
    "2-methoxyethanol": 17.2,
    "2-methyl-1-propanol": 16.777,
    "2-methyl-2-propanol": 12.47,
    "2-methylpentane": 1.89,
    "2-methylpyridine": 9.9533,
    "2-nitropropane": 25.654,
    "2-octanone": 9.4678,
    "2-pentanone": 15.2,
    "2-propanol": 19.264,
    "2-propen-1-ol": 19.011,
    "3-methylpyridine": 11.645,
    "3-pentanone": 16.78,
    "4-heptanone": 12.257,
    "4-methyl-2-pentanone": 12.887,
    "4-methylpyridine": 11.957,
    "5-nonanone": 10.6,
    "aceticacid": 6.2528,
    "acetophenone": 17.44,
    "a-chlorotoluene": 6.7175,
    "anisole": 4.2247,
    "benzaldehyde": 18.22,
    "benzonitrile": 25.592,
    "benzylalcohol": 12.457,
    "bromobenzene": 5.3954,
    "bromoethane": 9.01,
    "bromoform": 4.2488,
    "butanal": 13.45,
    "butanoicacid": 2.9931,
    "butanone": 18.246,
    "butanonitrile": 24.291,
    "butylamine": 4.6178,
    "butylethanoate": 4.9941,
    "carbondisulfide": 2.6105,
    "cis-1,2-dimethylcyclohexane": 2.06,
    "cis-decalin": 2.2139,
    "cyclohexanone": 15.619,
    "cyclopentane": 1.9608,
    "cyclopentanol": 16.989,
    "cyclopentanone": 13.58,
    "decalin-mixture": 2.196,
    "dibromoethane": 7.2273,
    "dibutylether": 3.0473,
    "diethylamine": 3.5766,
    "diethylsulfide": 5.723,
    "diiodomethane": 5.32,
    "diisopropylether": 3.38,
    "dimethyldisulfide": 9.6,
    "diphenylether": 3.73,
    "dipropylamine": 2.9112,
    "e-1,2-dichloroethene": 2.14,
    "e-2-pentene": 2.051,
    "ethanethiol": 6.667,
    "ethylbenzene": 2.4339,
    "ethylethanoate": 5.9867,
    "ethylmethanoate": 8.331,
    "ethylphenylether": 4.1797,
    "fluorobenzene": 5.42,
    "formamide": 108.94,
    "formicacid": 51.1,
    "hexanoicacid": 2.6,
    "iodobenzene": 4.547,
    "iodoethane": 7.6177,
    "iodomethane": 6.865,
    "isopropylbenzene": 2.3712,
    "m-cresol": 12.44,
    "mesitylene": 2.265,
    "methylbenzoate": 6.7367,
    "methylbutanoate": 5.5607,
    "methylcyclohexane": 2.024,
    "methylethanoate": 6.8615,
    "methylmethanoate": 8.8377,
    "methylpropanoate": 6.0777,
    "m-xylene": 2.3478,
    "n-butylbenzene": 2.36,
    "n-decane": 1.9846,
    "n-dodecane": 2.006,
    "n-hexadecane": 2.0402,
    "n-hexane": 1.8819,
    "nitrobenzene": 34.809,
    "nitroethane": 28.29,
    "n-methylaniline": 5.96,
    "n-methylformamide-mixture": 181.56,
    "n,n-dimethylacetamide": 37.781,
    "n,n-dimethylformamide": 37.219,
    "n-nonane": 1.9605,
    "n-octane": 1.9406,
    "n-pentadecane": 2.0333,
    "n-pentane": 1.8371,
    "n-undecane": 1.991,
    "o-chlorotoluene": 4.6331,
    "o-cresol": 6.76,
    "o-dichlorobenzene": 9.9949,
    "o-nitrotoluene": 25.669,
    "o-xylene": 2.5454,
    "pentanal": 10.0,
    "pentanoicacid": 2.6924,
    "pentylamine": 4.201,
    "pentylethanoate": 4.7297,
    "perfluorobenzene": 2.029,
    "p-isopropyltoluene": 2.2322,
    "propanal": 18.5,
    "propanoicacid": 3.44,
    "propanonitrile": 29.324,
    "propylamine": 4.9912,
    "propylethanoate": 5.5205,
    "p-xylene": 2.2705,
    "pyridine": 12.978,
    "sec-butylbenzene": 2.3446,
    "tert-butylbenzene": 2.3447,
    "tetrachloroethene": 2.268,
    "tetrahydrothiophene-s,s-dioxide": 43.962,
    "tetralin": 2.771,
    "thiophene": 2.727,
    "thiophenol": 4.2728,
    "trans-decalin": 2.1781,
    "tributylphosphate": 8.1781,
    "trichloroethene": 3.422,
    "triethylamine": 2.3832,
    "xylene-mixture": 2.3879,
    "z-1,2-dichloroethene": 9.2,
}
NUM_PROCESSES = 1
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
print(f" --- START OF LOG ---")
print(f" --- {timestamp} ---")

# --- SMILES to geometry ---
def smiles_to_geometry(smiles):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    AllChem.MMFFOptimizeMolecule(mol)
    atoms = mol.GetAtoms()
    conf = mol.GetConformer()
    return '\n'.join(
        f"{atom.GetSymbol()} {conf.GetAtomPosition(i).x:.6f} {conf.GetAtomPosition(i).y:.6f} {conf.GetAtomPosition(i).z:.6f}"
        for i, atom in enumerate(atoms)
    )

# --- DFT Thermo Calculation ---
def optimize_and_thermo(atom_block, temperature, dielectric=None):
    mol = gto.Mole()
    mol.atom = atom_block
    mol.basis = BASIS
    mol.charge = 0
    mol.spin = 0
    mol.unit = 'Angstrom'
    mol.build()

    mf = dft.RKS(mol)

    mol_optimized = geometric_solver.optimize(mf)

    if dielectric is None:
        mf_optimized = dft.RKS(mol_optimized)
        mf_optimized.xc = METHOD
    else:
        mf_optimized = mol_optimized.RKS(xc=METHOD).PCM()
        mf_optimized.with_solvent.method = 'IEF-PCM' # C-PCM, SS(V)PE, COSMO
        mf_optimized.with_solvent.eps = dielectric

    mf_optimized.kernel()

    if mf_optimized.converged:
        mo_energies = mf_optimized.mo_energy
        dipole_vector = mf_optimized.dip_moment()
        total_energy = mf_optimized.e_tot
    else:
        print("DFT geometry did not converge!")
        raise RuntimeError("DFT geometry did not converge!")

    if mo_energies is not None and dipole_vector is not None:
            homo = max(mo_energies[mo_energies < 0])
            lumo = min(mo_energies[mo_energies > 0])
            dipole = np.linalg.norm(dipole_vector)
            atoms = [atom[0] for atom in mol_optimized._atom]
            coords = mol_optimized.atom_coords()
            optimized_xyz = "\n".join(
                f"{atom} {x:.6f} {y:.6f} {z:.6f}"
                for atom, (x, y, z) in zip(atoms, coords)
            )

    print(" --- DFT RESULTS AFTER OPTIMISATION --- \n",
                f'HOMO (Hartrees) {homo}',
                f'LUMO (Hartrees) {lumo}',
                f'Dipole Moment (Debye) {dipole}',
                f'Total Energy (Hartrees) {total_energy}',
                f'Optimized XYZ {optimized_xyz}')

    # Compute nuclear Hessian
    hessian_matrix = mf.Hessian().kernel()

    # Frequency analysis
    freq_info = thermo.harmonic_analysis(mf.mol, hessian_matrix)
    print(" --- VIBRTAIONAL FREQUENCIES --- \n", freq_info)

    # Thermochemistry analysis at specified temperature and 1 atm
    thermo_info = thermo.thermo(mf, freq_info['freq_au'], temperature, pressure=101325)
    print(" --- THERMOCHEMISTRY --- \n", thermo_info)

    return thermo_info['G_tot']  # Return thermal Gibbs Free Energy

# --- Worker Function ---
def process_task(args):
    smiles, solvent_name, dielectric, temp = args
    try:
        print(f"Processing: {smiles} in {solvent_name} at {temp} K")
        geometry = smiles_to_geometry(smiles)
        g_vac = optimize_and_thermo(geometry, temp, dielectric=None)
        g_solv = optimize_and_thermo(geometry, temp, dielectric=dielectric)
        delta_g = g_solv[0] - g_vac[0]
        print(f"SUCCESS: {smiles} in {solvent_name} at {temp} K — ΔG = {delta_g:.4f} kcal/mol")
        return {
            "SMILES": smiles,
            "Solvent": solvent_name,
            "Temperature (K)": temp,
            "ΔG_solv (kcal/mol)": round(delta_g, 4)
        }
    except Exception as e:
        print(f"ERROR: {smiles} in {solvent_name} at {temp} K — {e}")
        return None

# --- Streamlit UI ---
st.title("🔬 Solvation Free Energy Calculator (PySCF + Streamlit)")
st.markdown("This app calculates solvation free energies using B3LYP/def2-SVP with PySCF in vacuum and solvent using ddCOSMO.")

smiles_input = st.text_area("Enter SMILES strings (one per line)", "CCO\nCC(=O)O\nCCN")
selected_solvents = st.multiselect("Select Solvents", list(DIELECTRIC_MAP.keys()), default=["dichloromethane", "tetrahydrofuran"])
run_button = st.button("Run Calculations")

if run_button:
    smiles_list = [s.strip() for s in smiles_input.splitlines() if s.strip()]
    job_args = [
        (smiles, solvent, DIELECTRIC_MAP[solvent], temp)
        for smiles in smiles_list
        for solvent in selected_solvents
        for temp in TEMPERATURES
    ]

    with st.spinner(f"Running {len(job_args)} calculations using {NUM_PROCESSES} processes..."):
        with Pool(NUM_PROCESSES) as pool:
            results = pool.map(process_task, job_args)

    results = [r for r in results if r]
    df = pd.DataFrame(results)

    if not df.empty:
        st.success("✅ Calculations complete.")
        st.dataframe(df)

        csv = df.to_csv(index=False).encode()
        st.download_button("📥 Download Results CSV", csv, f"solvation_results_{timestamp}.csv", "text/csv")
