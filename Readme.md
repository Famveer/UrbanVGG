# UrbanVGG

Explaining urban safety perception through visual features. [Paper](https://fmorenovr.github.io/documents/papers/book_chapters/2021_ICIC.pdf).

# Requirements

- **Python**>=3.12

# Installation
```
  pip install -r requirements.txt
```

# Data

Obtain the Place Pulse 1.0 dataset [here](https://drive.google.com/drive/folders/1R1AcUP9IN2IwyY280v98vtJBS17N3IxJ?usp=sharing).

### Data Preparation

* Download images and `scores.csv`.  
* Create a `.env` file, and add the path of the data downloaded and models.  
  ```
    DATA_PATH=/path_to/datasets/
    MODEL_PATH=/path_to/models/
  ```
* First, run the notebook `notebooks/Data/Organize_Information.ipynb`.  
  Then, run the notebook `notebooks/Data/Statistics.ipynb`.  
  Next, run the notebook `notebooks/Data/Feature_Extraction.ipynb`.  

* Train models running `notebooks/Models/`.  
  Then, run explanations `notebooks/Explanations/`.  

# Citation

```
@inproceedings{moreno2021understanding,
  title={Understanding safety based on urban perception},
  author={Moreno-Vera, Felipe},
  booktitle={International Conference on Intelligent Computing},
  pages={54--64},
  year={2021},
  organization={Springer}
}
```

# Contact us  
For any issue please kindly email to `felipe [dot] moreno [at] fgv [dot] br`
