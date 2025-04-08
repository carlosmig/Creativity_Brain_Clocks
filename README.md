
# Brain Age Gap (BAG) Analysis

This project analyzes brain age gaps (BAGs) across various participant groups using Support Vector Machine (SVM) models and functional connectivity matrices. The code loads and processes connectivity data, calculates group-wise metrics, and generates visualizations of results.

## Requirements

### Python Version
The code is compatible with Python 3.x.

### Required Libraries
To run this code, the following Python libraries are required. You can install them via `pip`:

```bash
pip install numpy matplotlib scipy scikit-learn statsmodels nilearn nibabel wordcloud
```

Here’s a breakdown of each library used:

- **numpy**: For numerical operations and matrix manipulation.
- **matplotlib**: For generating plots and visualizations.
- **scipy**: For statistical analysis, including Pearson correlations and linear regression.
- **scikit-learn**: For machine learning models (SVM) and cross-validation.
- **statsmodels**: For multiple testing correction (Benjamini-Hochberg).
- **nilearn**: For neuroimaging data manipulation.
- **nibabel**: For handling neuroimaging data formats (e.g., Nifti).
- **wordcloud**: For generating word clouds in result visualizations.

Additionally, the `bct` library is used for brain connectivity measures, and `plot_violins` is used for custom violin plots. Ensure you have these libraries available in your working environment:
- **bctpy**: Brain connectivity toolbox in Python. Install using:
  ```bash
  pip install bctpy
  ```
- **plot_violins**: A custom module assumed to be available in your working directory.
- **networkx**: required for using bctpy. Install using:
  ```bash
  pip install networkx
  ```


## File Structure
The data files must be organized as follows:

```
.
├── Training_SVMs_Data/
│   ├── FCs_training_augmented.npy
│   └── ages_training_augmented.npy
├── Gaming/
│   ├── ages_SC1.npy
│   ├── ages_SC2.npy
│   ├── FCs_SC1.npy
│   ├── FCs_SC2.npy
│   ├── SCs_SC1.npy
│   ├── SCs_SC2.npy
│   └── playing_time.npy
├── Tango/
│   ├── ages_high_tango.npy
│   ├── ages_low_tango.npy
│   ├── FCs_high_tango.npy
│   ├── FCs_low_tango.npy
│   ├── hours_high_tango.npy
│   └── hours_low_tango.npy
├── Visual/
│   ├── ages_nonvisual.npy
│   ├── ages_visual.npy
│   ├── experience_visual.npy
│   ├── FCs_nonvisual.npy
│   └── FCs_visual.npy
├── Musicians/
│   ├── ages_musicians.npy
│   ├── ages_nonmusicians.npy
│   ├── FCs_musicians.npy
│   ├── FCs_nonmusicians.npy
│   └── years_music.npy
├── Learning/
│   ├── ages_sonata.npy
│   ├── APM_post.npy
│   ├── APM_pre.npy
│   ├── FCs_sonata_post.npy
│   └── FCs_sonata_pre.npy
├── maps_groups/
│   ├── Ds_corrs.npy
│   ├── Ds_experts.npy
│   └── Ds_training.npy
└── neurosynth/
    ├── parcellated_data.npy
    └── cognitive_terms.npy
```

## Usage

1. Clone this repository and navigate to the project directory.
2. Ensure you have the necessary data files in the correct structure as shown above.
3. Run the main script:
   ```bash
   python main_script.py
   ```

The script will output various statistical results and visualizations showing brain age gap analysis across groups.


