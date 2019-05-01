A crafty implementation of Google's [MorphNet](https://arxiv.org/abs/1711.06798) (and derivative iterations) in PyTorch.

This API is undergoing wild changes as it approaches release.
* Almost every change will be a breaking change.
* __Do not__ rely on the functions as they currently are

Please feel free to look around, but bookmark which release tag it was. Master will be changing, viciously.

It is recommended that you consult [the current working branch](https://github.com/stephenjfox/Morph.py/tree/release/v0.1.0) for a more realistic view of the codebase


---

# Understanding [MorphNet](https://arxiv.org/abs/1711.06798)

A Stephen Fox endeavor to become an Applied AI Scientist.

## Background Resources

### Key Ideas

1. Make it simple to refine neural architectures
2. Focus on dropping model parameter size while __keeping performance as high as possible__
3. Make the tools user-friendly, and clearly documented

### Project Roadmap

- Please see [the GitHub Project board](https://github.com/stephenjfox/Morph.py/projects/1)

---

## Usage

### Installation

`pip install morph-py`

### Code Example

```python
import morph

morph_optimizer = None
# train loop
for e in range(epoch_count):

  for input, target in dataloader:
    optimizer.zero_grad() # optional: zero gradients or don't...
    output = model(input)

    loss = loss_fn(output, target)
    loss.backward()
    optim.step()


    # setup for comparing the morphed model
    if morph_optimizer:
      morph_optimizer.zero_grad()
      morph_loss = loss_fn(morph_model(input), target)

      logging.info(f'Morph loss - Standard loss = {morph_loss - loss}')

      morph_loss.backward()
      morph_optimizer.step()


    # Experimentally supported: Initialize our morphing halfway training
    if e == epoch_count // 2:
      # if you want to override your model
      model = morph.once(model)

      # if you want to compare in parallel
      morph_model = morph.once(model)

      # either way, you need to tell your optimizer about it
      morph_optimizer = init_optimizer(params=morph_model.parameters())
      
```

## What is Morph.py?

Morph.py is a Neural Network Architecture Optimization toolkit targeted at Deep Learning researchers
  and practitioners.
* It acts outside of the current paradigm of [Neural Architecture Search](https://github.com/D-X-Y/awesome-NAS)
  while still proving effective
* It helps one model accuracy of a model with respect to its size (as measured by "count of model parameters")
  * Subsequently, you could be nearly as effective (given some margin of error) with a __much__ smaller
    memory footprint
* Provides you, the researcher, with [better insight on how to improve your model](https://github.com/stephenjfox/Morph.py/projects/3)

Please enjoy this [Google Slides presentation](https://goo.gl/ZzZrng)

Coming soon:
* A walkthrough of the presentation (more detail than my presenter's notes)
* More [supported model architectures](https://github.com/stephenjfox/Morph.py/projects/2)


### Current support

* Dynamic adjustment of a given layer's size
* Weight persistence across layer resizing
  * To preserve all the hard work you spent in

---

# Contributing

## Setup (to work alongside me)

`git clone https://github.com/stephenjfox/Morph.py.git`

### Requisites

#### [Install Anaconda](https://www.anaconda.com/download/)
* They've made it easier with the years. If you haven't already, please give it a try

#### Install Pip

1. `conda install pip`
2. Proceed as normal

### Dependencies

- Jupyter Notebook
  * And a few tools to make it better on your local environment like `nb_conda`, `nbconvert`, and `nb_conda_kernels`
- Python 3.6+ because [Python 2 is dying](https://pythonclock.org/)
- PyTorch (`conda install torch torchvision -c pytorch`)

All of these and more are covered in the `environment.yml` file:
+ Simply run `conda env create -f environment.yml -n <desired environment name>`