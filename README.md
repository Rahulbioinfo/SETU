git clone https://github.com/Rahulbioinfo/SETU.git
cd SETU

conda create -n setu python=3.10 -y
conda activate setu

conda install -c conda-forge rdkit=2022.09.5 -y

pip install numpy==1.26.4 \
pandas==2.3.3 \
scikit-learn==1.7.2 \
xgboost==1.7.6 \
joblib==1.5.3 \
shap==0.49.1 \
mordred==1.2.0



Example Workflow
# Activate environment
conda activate setu

# Enter SETU directory
cd SETU

# Run screening
python screen_mgtbind_ligand_qsar.py
