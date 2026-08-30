
Creative Experiences and Brain Clocks
===========================================

Coronel-Oliveros, C., Migeot, J., Lehue, F., Amoruso, L., Kowalczyk-Grębska, N., Jakubowska, N., Mandke, K. N., ... Ibanez, A. (2025).
Creative experiences and brain clocks. Nature Communications.


Overview
--------

This repository contains the code, data organization, and materials for the study "Creative Experiences and Brain Clocks", which investigates how different creative practices (e.g., music, dance, gaming, visual arts) influence brain aging and cognitive resilience using neuroimaging, machine learning, and neurocognitive mapping tools.

We developed and tested brain age prediction models, spin-based surrogate testing, functional connectivity analyses, and cognitive decoding, combining these with behavioral data on creative expertise and training interventions.

Individual data related to music expertise design is available upon request and was not included here due to GDPR regulations.

## Repository Structure

- **Gaming/** – Data for the gaming group.
- **Global_coupling/** – Global coupling model and parameter files.
- **Learning/** – Pre/post training data and actions-per-minute analysis (APM).
- **Tango/** – Data for tango dancers.
- **Visual/** – Data for visual artists.
- **Training_SVMs_Data/** – Data matrices and labels for machine learning models.
- **neurosynth_spin_test/** – Code and files for cognitive decoding and spin tests:
  - `AAL_coordinates.txt` – Region coordinates used for distance matrix.
  - `parcellated_data.npy` – Neurosynth associations maps per term.
  - `cognitive_terms.npy` – List of cognitive term labels.
  - `Ds_files` – Different brain maps for spatial autocorrelations.
- **Main_Script.py** – Main pipeline script to reproduce figures and analyses.
- **params_SVM.npy** – Saved hyperparameters for support vector machine models.
- **experts.svg** – Word cloud visualization of top cognitive correlations for experts.
- **training.svg** – Word cloud visualization of top cognitive correlations for the training group.
- **plot_violins.py** – Customized violin plots. 
- **README.md** – This file.

## Requirements

### Standard Python Libraries

- Python 3.9+
- NumPy
- SciPy
- scikit-learn
- matplotlib
- seaborn
- statsmodels
- nilearn

### Additional Libraries

- netneurotools: pip install netneurotools. https://netneurotools.readthedocs.io/en/latest/
- wordcloud: pip install wordcloud. https://pypi.org/project/wordcloud/
- brainsmash: pip install brainsmash. https://brainsmash.readthedocs.io/en/latest/
- neuromaps: pip install neuromaps. https://pypi.org/project/neuromaps/
- surfplot: pip install surfplot. https://pypi.org/project/surfplot/
- bctpy: pip install bctpy. https://pypi.org/project/bctpy/
- networkx: pip install networkx. https://pypi.org/project/networkx/
- nibabel: pip install nibabel. https://pypi.org/project/nibabel/

### Running the Project

1. Clone the repository:
   git clone https://github.com/<your-org-or-username>/Creativity_Brain_Clocks.git
   cd Creativity_Brain_Clocks
   
2. Run the main script to generate plots:
   python Main_Script.py

Key Features
------------

- Brain Age Prediction: Models estimating brain age using functional connectivity data across creative domains.
- Group Comparisons: Contrasts between experts and non-experts in music, dance, gaming, and visual arts.
- Training Effects: Longitudinal assessment of training (e.g., Sonata project) on brain age gaps.
- Cognitive Decoding: Mapping neural data onto cognitive ontologies using spin tests and surrogate null models.
- Visualization: Word clouds, violin plots, cortical projections using surface-based mapping (e.g., FsLR surfaces).


Citation
--------

If you use this repository, please cite:

Coronel-Oliveros, C., Migeot, J., Lehue, F., Amoruso, L., Kowalczyk-Grębska, N., Jakubowska, N., Mandke, K. N., ... Ibanez, A. (2025).
Creative experiences and brain clocks. Nature Communications.

Contact
-------

For questions or collaboration inquiries, please contact:
Carlos Coronel-Oliveros
📧 carlos.coronel@gbhi.org

