
Creative Experiences and Brain Health Clocks
===========================================

Coronel-Oliveros, C., Migeot, Lehue, F. J., Amoruso, L., Kowalczyk-Grębska, N., Jakubowska, N., Mandke, K. N., ... Ibanez, A. (2025).
Creative experiences and brain health clocks. Nature Communications.


Overview
--------

This repository contains the code, data organization, and materials for the study "Creative Experiences and Brain Health Clocks", which investigates how different creative practices (e.g., music, dance, gaming, visual arts) influence brain aging and cognitive resilience using neuroimaging, machine learning, and neurocognitive mapping tools.

We developed and tested brain age prediction models, spin-based surrogate testing, functional connectivity analyses, and cognitive decoding, combining these with behavioral data on creative expertise and training interventions.


Repository Structure
--------------------

.
├── Gaming/                  # Data and analysis scripts for the gaming group
├── Global_coupling/        # Global coupling parameter files
├── Learning/               # Pre/post training data and actions-per-minute analysis
├── Musicians/              # Data and scripts for the music expert group
├── Tango/                  # Data and scripts for tango dancers
├── Visual/                 # Data and scripts for visual artists
├── Training_SVMs_Data/     # Data matrices and labels for ML models
├── neurosynth_spin_test/   # Code and data for cognitive decoding and spin test
│   ├── AAL_coordinates.txt
│   ├── parcellated_data.npy
│   └── cognitive_terms.npy
├── Main_Script.py          # Main script to reproduce key figures and results
├── params_SVM.npy          # Saved parameters for the SVM models
├── experts.svg             # Word cloud visualization for expert group correlations
├── training.svg            # Word cloud visualization for training-related terms
├── README.md               # This file


Getting Started
---------------

### Requirements

- Python 3.9+
- NumPy
- SciPy
- scikit-learn
- matplotlib
- seaborn
- statsmodels
- nilearn
- netneurotools
- wordcloud
- brainsmash
- neuromaps
- surfplot
- bctpy
- nibabel

### Running the Project

1. Clone the repository:
   git clone https://github.com/<your-org-or-username>/brain-health-clocks.git
   cd brain-health-clocks

2. (Optional) Create a virtual environment:
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   pip install -r requirements.txt

3. Run the main script to generate plots:
   python Main_Script.py


Key Features
------------

- Brain Age Prediction: Models estimating brain age using functional connectivity data across creative domains.
- Group Comparisons: Contrasts between experts and non-experts in music, dance, gaming, and visual arts.
- Training Effects: Longitudinal assessment of training (e.g., Sonata project) on brain health metrics.
- Cognitive Decoding: Mapping neural data onto cognitive ontologies using spin tests and surrogate null models.
- Visualization: Word clouds, violin plots, cortical projections using surface-based mapping (e.g., FsLR surfaces).


Citation
--------

If you use this repository, please cite:

Coronel-Oliveros, C., Migeot, Lehue, F. J., Amoruso, L., Kowalczyk-Grębska, N., Jakubowska, N., Mandke, K. N., ... Ibanez, A. (2025).
Creative experiences and brain health clocks. Nature Communications.

License
-------

This project is released under the MIT License. See LICENSE for details.


Contact
-------

For questions or collaboration inquiries, please contact:
Carlos Coronel-Oliveros
📧 carlos.coronel@gbhi.org

