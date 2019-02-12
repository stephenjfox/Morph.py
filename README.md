A crafty implementation of Google's [MorphNet](https://arxiv.org/abs/1711.06798) (and derivative iterations) in PyTorch.

This API is undergoing wild changes as it approaches release.
* Almost every change will be a breaking change.
* __Do not__ rely on the functions as they currently are

Please feel free to look around, but bookmark which release tag it was. Master will be changing, viciously.


---

# Understanding [MorphNet](https://arxiv.org/abs/1711.06798)

A Stephen Fox endeavor to become an Applied AI Scientist.

## Setup (to work alongside me)

`git clone https://github.com/stephenjfox/Morph.py.git`

## Requisites

### [Install Anaconda](https://www.anaconda.com/download/)
* They've made it easier with the years. If you haven't already, please give it a try

### Install Pip

1. `conda install pip`
2. Proceed as normal

### Dependencies

- Jupyter Notebook
  * And a few tools to make it better on your local environment like `nb_conda`, `nbconvert`, and `nb_conda_kernels`
- Python 3.6+ because [Python 2 is dying](https://pythonclock.org/)
- PyTorch (`conda install torch torchvision`)
