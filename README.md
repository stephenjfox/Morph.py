A crafty implementation of Google's [MorphNet](https://arxiv.org/abs/1711.06798) (and derivative iterations) in PyTorch.

This API is undergoing wild changes as it approaches release.
* Almost every change will be a breaking change.
* __Do not__ rely on the functions as they currently are

Please feel free to look around, but bookmark which release tag it was. Master will be changing, viciously.

It is recommended that you consult [the current working branch](https://github.com/stephenjfox/Morph.py/tree/release/v0.1.0) for a more realistic view of the codebase

---

## Update: State of the Project, May 2019

With the recent hype around MorphNet (thanks to [Google's blog post]) I've received emails and new GitHub
  issues about this project's usability. Allow me to address those here.
1. This project was started in late-January 2019 to serve as a toolkit to MorphNet's functionality
    * It was a part of my research fellowship, but unfortunately faded to the background.
    * As such, most of the automatic network rearchitecting isn't currently in _this_ codebase.
    * For the next two months (May, June 2019) I will be intermittently incorporating my private usage
      of the tools publicly available herein, that achieve the hype of the paper
2. This project __is__ intended to be a _clear_ and _legible_ implementation of the algorithm
    * After seeing the original codebase, I was dismayed at how difficult the code was to follow
      * Admittedly, I'm not the world's expert on Tensorflow/PyTorch code.
      * But I know clever code isn't instructive
    * As such, I wanted a simpler approach, that someone could step through and __at any point__ trace the
      code back to the paper's ideas
    * This would allow the simplicity of the algorithm to be simply manifest to the user
3. Open-Source != "Will work for free"
    * Contributions are preferable to complaints. I can more easily take 20 minutes to read your PR than I
      can write a mission statement (like this one) or fend off gripes-by-email.
      * Instead of emailing me, create a descriptive issue of what you want - with an idealized code example -
        or of what is broken.
    * We have the [seeds of a conversation started](https://github.com/stephenjfox/Morph.py/issues/14) on how
      to make Morph.py ready for primetime, but until then please be patient!
    * Hindsight is indeed 20-20: had I known certain opportunities wouldn't pan out, or that Google would drop
      a bomb in my lap, I wouldn't have been caught off guard for the rush of new users.

If you want to submit a PR for a decent issue/complaint/feature request template, that would be much appreciated.

Thank you for your patience and I hope you enjoy what I have here for you at present.

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

### Code Example: Using the tools provided

The following code example is the _toolkit_ use case.
* __This is not__ the auto-magical, "__make my network better for free__" path
* __This is__ how you could manually pick-and-choose when morphing happened to your network
  * Maybe you know that there's a particularly costly convolution
  * Or you know that your RNN can hit the exploding gradient problem around epoch 35
  * Or any other use case (FYI: the above can [maybe _should_] be solved by other means)

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

### Code Example: Automatic morphing of your architecture

__TODO__

Notes:
* This is more like what Google promised regarding improved performance.
* This project focuses on __model size regularization__ and __avoids FLOPs__ regularization.

```python
# TODO: Code example of the dynamic, automatic morphing implementation
```


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


<!-- Links -->

[Google's blog post]: https://ai.googleblog.com/2019/04/morphnet-towards-faster-and-smaller.html
