# Automatic Regional Quality Estimation for Echocardiography (arqee)

## Citation
You should cite the following paper when using the code in this repository:

Van De Vyver, Gilles, et al. "Regional quality estimation for echocardiography using deep learning." arXiv preprint arXiv:2408.00591 (2024). https://arxiv.org/abs/2408.00591


## Toturial
Jupyter notebook toturials are available at

- End to end quality prediction: [tutorial/tutorial_end_to_end_quality_prediction.ipynb](tutorial/tutorial_end_to_end_quality_prediction.ipynb).

- Combine gCNR with segmentation: [tutorial/tutorial_gcnr.ipynb](tutorial/tutorial_gcnr.ipynb)

## Installation
The package can be installed using pip from github:
```bash
pip install https://github.com/GillesVanDeVyver/arqee.git
```
Alternatively, you can download the source code and install it using pip:
```bash
pip install .
```

## Main features

- Calculate quality per myocardial region for apical views via end to end (black-box) model
- Get quality measure based on gCNR per myocardial region for apical views

![tutorial/image_quality_prediction.gif](tutorial/image_quality_prediction.gif)



## Contact

gillesvandevyver@hotmail.be <br />
https://www.linkedin.com/in/gilles-van-de-vyver/

Feature requests and feedback are welcome.


The package has been developed using python 3.10.
