**Ribosome Phenotypes Enable Rapid Antibiotic Susceptibility Testing in Escherichia coli**

This github repo provides the code necessary to generate image datasets and classify _E. coli_ by ribosome phenotype. For image segmentation and bacterial segmentation statistics, see https://github.com/piedrro/napari-bacseg.

Data is available on Zenodo: https://zenodo.org/records/11656505

After cloning the repo, the project can be installed with the pyproject.toml file.

- main.py: runs classification and model training; choose a task_name to load the required metadata for each model
- data/: contains .csv metadata for each model
- src/ribophene/: contains scripts for data loading, model training, and visualization
- scripts/: internal code for image generation
