# PPI-GCN
## Protein-Protein-Interaction-Sites-Prediction-By-GCN
This project implements creating a GCN graph to predict the missing sites of Protein-Protein Interactions. <br>
Three datasets are used in this implementation (Dset 186, Dset 72, and PDBset 164). <br>
Three features are axtracted: <br>
1PSSM: PSI-BLAST
2Secondary Structure: DSSP Library
3Raw Protein Sequence. 

The implementation requires the following packages to be installed: 
- Pytorch
pip install pytorch
- Graphein Libraray: 
pip install graphein
- DSSP:
conda install -c salilab dssp
- Pymol:
conda install -c schrodinger pymol 
- GetContacts
conda install -c conda-forge vmd-python
git clone https://github.com/getcontacts/getcontacts
- Add folder to PATH
echo "export PATH=\$PATH:`pwd`/getcontacts" >> ~/.bashrc
source ~/.bashrc
To test the installation, run:

cd getcontacts/example/5xnd
get_dynamic_contacts.py --topology 5xnd_topology.pdb \
                        --trajectory 5xnd_trajectory.dcd \
                        --itypes hb \
                        --output 5xnd_hbonds.tsv
