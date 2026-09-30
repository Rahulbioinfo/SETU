import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import Descriptors
from mordred import Calculator, descriptors

import ligand_preprocessor_loader

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = BASE_DIR / "mgtbind_non_degraders.sdf"

MODEL_FILE = (
    BASE_DIR
    / "optimized_xgboost_pipeline.pkl"
)

OUTPUT_DIR = (
    BASE_DIR
    / "External_Screening"
    / "MGTBind"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_ALL = OUTPUT_DIR / "MGTBind_All_Predictions.csv"
OUTPUT_TOP1000 = OUTPUT_DIR / "MGTBind_Top1000.csv"
OUTPUT_TOP100 = OUTPUT_DIR / "MGTBind_Top100.csv"
OUTPUT_TOP50 = OUTPUT_DIR / "MGTBind_Top50.csv"
OUTPUT_SUMMARY = OUTPUT_DIR / "MGTBind_Screening_Summary.csv"

sys.modules[
    "ligand_qsar_hyperparameter_optimization_resumable"
] = ligand_preprocessor_loader

print("=" * 80)
print("EXTERNAL MGTBIND LIGAND QSAR SCREENING")
print("=" * 80)

print(f"Input : {INPUT_FILE}")
print(f"Model : {MODEL_FILE}")
print(f"Output: {OUTPUT_DIR}")

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"MGTBind SDF not found:\n{INPUT_FILE}"
    )

if not MODEL_FILE.exists():
    raise FileNotFoundError(
        f"Model not found:\n{MODEL_FILE}"
    )

start_time = time.time()

print("\nLoading trained ligand XGBoost model...")

model = joblib.load(
    MODEL_FILE
)

preprocessor = model.named_steps[
    "preprocessor"
]

xgb = model.named_steps[
    "model"
]

required_features = list(
    preprocessor.columns_
)

print(
    f"Model descriptors required: "
    f"{len(required_features)}"
)

print(
    f"XGBoost features: "
    f"{xgb.n_features_in_}"
)

print("\nReading MGTBind SDF...")

supplier = Chem.SDMolSupplier(
    str(INPUT_FILE),
    removeHs=False,
    sanitize=True
)

molecules = []

for index, mol in enumerate(
    supplier,
    start=1
):

    if mol is None:
        print(
            f"Warning: invalid molecule at SDF record "
            f"{index}"
        )
        continue

    smiles = Chem.MolToSmiles(
        mol,
        canonical=True,
        isomericSmiles=True
    )

    properties = {}

    for name in mol.GetPropNames():

        try:
            properties[name] = mol.GetProp(name)
        except Exception:
            properties[name] = ""

    if "Ligand" in properties:
        ligand_id = properties["Ligand"]

    elif "Name" in properties:
        ligand_id = properties["Name"]

    elif mol.HasProp("_Name"):
        ligand_id = mol.GetProp("_Name")

    else:
        ligand_id = f"MGTBind_{index:06d}"

    molecules.append(
        {
            "MGTBind_ID": ligand_id,
            "SMILES": smiles,
            "SDF_Record": index,
            "Properties": properties,
            "Mol": mol
        }
    )

print(
    f"Valid molecules: "
    f"{len(molecules):,}"
)

if len(molecules) == 0:
    raise ValueError(
        "No valid molecules were found in the SDF."
    )

print("\nGenerating Mordred descriptors...")

calculator = Calculator(
    descriptors,
    ignore_3D=True
)

mols = [
    item["Mol"]
    for item in molecules
]

mordred_table = calculator.pandas(
    mols
)

mordred_table.columns = [
    str(column)
    for column in mordred_table.columns
]

print(
    f"Generated Mordred matrix: "
    f"{mordred_table.shape[0]:,} × "
    f"{mordred_table.shape[1]:,}"
)

print("\nGenerating RDKit descriptors...")

rdkit_rows = []

for mol in mols:

    rdkit_rows.append(
        {
            "MolWt": Descriptors.MolWt(mol),
            "LogP": Descriptors.MolLogP(mol),
            "TPSA": Descriptors.TPSA(mol),
            "HBD": Descriptors.NumHDonors(mol),
            "HBA": Descriptors.NumHAcceptors(mol),
            "RotBonds": Descriptors.NumRotatableBonds(mol),
            "RingCount": Descriptors.RingCount(mol),
            "AromaticRings": Descriptors.NumAromaticRings(mol)
        }
    )

rdkit_table = pd.DataFrame(
    rdkit_rows
)

print(
    f"RDKit matrix: "
    f"{rdkit_table.shape[0]:,} × "
    f"{rdkit_table.shape[1]:,}"
)

descriptor_table = pd.concat(
    [
        rdkit_table,
        mordred_table
    ],
    axis=1
)

missing_features = [
    feature
    for feature in required_features
    if feature not in descriptor_table.columns
]

if missing_features:

    print("\nMissing required descriptors:")

    for feature in missing_features:
        print(" ", feature)

    raise ValueError(
        f"{len(missing_features)} model descriptors "
        f"are missing."
    )

X = descriptor_table[
    required_features
].copy()

X = X.apply(
    pd.to_numeric,
    errors="coerce"
)

X = X.replace(
    [np.inf, -np.inf],
    np.nan
)

print(
    f"\nExact model matrix: "
    f"{X.shape[0]:,} × "
    f"{X.shape[1]}"
)

print("\nApplying trained preprocessing...")

X_transformed = preprocessor.transform(
    X
)

print(
    f"Transformed matrix: "
    f"{X_transformed.shape[0]:,} × "
    f"{X_transformed.shape[1]:,}"
)

print("\nGenerating QSAR predictions...")

probabilities = model.predict_proba(
    X
)[:, 1]

predictions = model.predict(
    X
)

results = pd.DataFrame(
    {
        "MGTBind_ID": [
            item["MGTBind_ID"]
            for item in molecules
        ],
        "SMILES": [
            item["SMILES"]
            for item in molecules
        ],
        "SDF_Record": [
            item["SDF_Record"]
            for item in molecules
        ],
        "LigandProbability": probabilities,
        "PredictedClass": predictions
    }
)

results = results.sort_values(
    "LigandProbability",
    ascending=False
).reset_index(
    drop=True
)

results.insert(
    0,
    "QSAR_Rank",
    np.arange(
        1,
        len(results) + 1
    )
)

results.to_csv(
    OUTPUT_ALL,
    index=False
)

results.head(1000).to_csv(
    OUTPUT_TOP1000,
    index=False
)

results.head(100).to_csv(
    OUTPUT_TOP100,
    index=False
)

results.head(50).to_csv(
    OUTPUT_TOP50,
    index=False
)

summary = pd.DataFrame(
    [
        {
            "Database": "MGTBind non-degraders",
            "Input_Molecules": len(results),
            "Predicted_Binders": int(
                np.sum(predictions == 1)
            ),
            "Mean_LigandProbability": float(
                np.mean(probabilities)
            ),
            "Median_LigandProbability": float(
                np.median(probabilities)
            ),
            "Maximum_LigandProbability": float(
                np.max(probabilities)
            ),
            "Top50_MeanProbability": float(
                results.head(50)[
                    "LigandProbability"
                ].mean()
            ),
            "Top100_MeanProbability": float(
                results.head(100)[
                    "LigandProbability"
                ].mean()
            ),
            "Top1000_MeanProbability": float(
                results.head(1000)[
                    "LigandProbability"
                ].mean()
            )
        }
    ]
)

summary.to_csv(
    OUTPUT_SUMMARY,
    index=False
)

elapsed = time.time() - start_time

print("\n" + "=" * 80)
print("MGTBIND SCREENING COMPLETED")
print("=" * 80)

print(
    f"Compounds screened : {len(results):,}"
)

print(
    f"Mean probability   : "
    f"{probabilities.mean():.6f}"
)

print(
    f"Maximum probability: "
    f"{probabilities.max():.6f}"
)

print(
    f"Predicted binders  : "
    f"{int(np.sum(predictions == 1)):,}"
)

print(
    f"Runtime            : "
    f"{elapsed / 60:.2f} minutes"
)

print("\nTop 20 compounds:")

print(
    results[
        [
            "QSAR_Rank",
            "MGTBind_ID",
            "LigandProbability",
            "PredictedClass",
            "SMILES"
        ]
    ].head(20).to_string(
        index=False
    )
)

print("\nOutput files:")

print(OUTPUT_ALL)
print(OUTPUT_TOP1000)
print(OUTPUT_TOP100)
print(OUTPUT_TOP50)
print(OUTPUT_SUMMARY)

print("\nDONE.")
