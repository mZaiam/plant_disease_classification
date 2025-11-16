# Plant Disease Classification

This repository contains the final project of the students Matheus Zaia Monteiro and Gustavo Uchôa Barros, at ILUM School of Science, under supervision of Professor Daniel Roberto Cassar. 

## Repository Organization

The repository is organized as follows:

- `plantvillage/`: the PlantVillage dataset used in this project;
- `digipathos/`: the Digipathos dataset used in this project;
- `data_exploration/`: the exploratory analysis over the data;
- `models/`: the pipelines of the project.

Inside the `models/` directory, it is possible to found 4 directories:
- `plantvillage/`;
- `feijao/`, `mandioca/` and `milho/`.
  
The former contains the CNN pre-trained on the PlantVillage dataset. The latter contains the models trained on the Digipathos data. Each one of these directories contains:
- `best_models/`: best models for each pipeline;
- `model_analysis.ipynb`: the confusion matrices and hypothesis tests for each pipeline;
- `model_explanation/`: the explainability analysis performed on the models. Methods such as `Occlusion`, `GradSHAP` and `GardCAM` were used.
