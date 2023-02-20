# Deep Learning project template
![Formatting and tests workflow](https://github.com/mkmenta/deep-learning-project-template/actions/workflows/main-action.yml/badge.svg)

Deep Learning base project template using the following stack
- [PyTorch](https://pytorch.org/)
- [PyTorch Lightning](https://www.pytorchlightning.ai/)
- [Neptune.ai](https://neptune.ai/)
- [Scikit Learn](https://scikit-learn.org/)

Go to the section [Reasoning](#reasoning) below to find out the why I took the decisions of implementing the template
in this way.

## Environment setup
Install the requirements to run the project. Note that most/all the versions are given by `>=` in order to show the 
specific versions used for development, but at the same time allow them to be updated.
```
python3 -m pip install -r requirements.txt
```

Install the requirements for code formatting and testing.
```
python3 -m pip install -r test-requirements.txt
```

Sign up at neptune.ai and export the following environment variables:
```bash
NEPTUNE_API_TOKEN="YOUR_NEPTUNE_API_TOKEN"
NEPTUNE_PROJECT_NAME=[YOUR_NEPTUNE_USER_NAME]/[YOUR_NEPTUNE_PROJECT_NAME]
```

## Run
Run
```
python3 -m project.run --help
```
to get help about the possible arguments.

### Train
Example of training command:
```
python3 -m project.run --exp-name trial --learning-rate 0.001
```

## Reasoning
This section explains the reasons why this template is implemented the way it is. 

### Why these libraries?
- **Why PyTorch?** I have been using PyTorch since 2018 and it has never let me down :) I would actually say the opposite.
- **Why PyTorch Lightning?** 
  - After implementing and extending several PyTorch projects I have found that a lot of my time
  was wasted working and fixing bugs on the same generic stuff (like epoch loops, dataloading, multi-gpu setup, etc.). 
  PyTorch lightning lets you skip all that while still being very flexible. 
  - It helps other people getting 
  into my projects: if the other person knows the PyTorch Lightning structure, that person will for sure save a lot of 
  time knowing where to look and how to read my pipeline.
  - The only problem I have found with PyTorch Ligthning is it has some default behaviors that are sometimes tricky to 
  understand and debug even with the docs, because they are not clearly specified from the beginning. For example: 
  the default logging, the default saving of checkpoints + folder structure and the accumulation of the values in the 
  *return* of each `**_step()` function. **In order to avoid all these issues check the list in the
  [PyTorch Lightning Section](#notes-about-pytorch-lightning)**
- **Why Neptune.ai?** It is very convenient to have all the experiments logs saved in the cloud for three main reasons:
  1) it makes it easier to collaborate with other people.
  2) it is safer (in my opinion) in case of disk failure or unintentional deletion of the data.
  3) it is more accessible (you don't need to run your own tensorboard server to see them).
  - Why Neptune.ai and not others? I have found myself working with neptune.ai and it has worked fine for me, so I
  haven't considered other options. Although there are many out there and they can be easily [integrated with PyTorch 
  Lightning](https://pytorch-lightning.readthedocs.io/en/stable/extensions/logging.html).
- **Why Scikit Learn?** It is a well known library for scientific projects with Python and I think that it is always 
better to first look for the implementations that are already out there (for example, of metric computations). Specially
if the library is stable and validated by the community because we can avoid many undesired bugs and inconsistencies.

### Why implement it like this?
Here are some explanations about the decision taken during the implementation. Other explanations can also be found as
comments in the code.
- The GitHub Actions allow to automatically test the code and to keep it clean.
- The project structure is quite intuitive. The only special thing is the separation between `main.py` and `solver.py`. 
  - `solver.py` 
  contains the `LightningModule`, which tends to become quite long. So, I isolated it from the main function and CLI into 
  a separate module.
  - `main.py` is intuitively the first script to check and the one containing the main function.
- The `val_**` and `test_**` functions in PytorchLightning tend to share a lot of the code in my experience. So, they have
been unified into new `eval_**` functions.
- Metrics:
  - The metrics are separated in standalone modules to isolate them in shorter files and make them easily re-usable in different projects. 
  - For better performance and efficiency, they are designed to compute a few partial values step by step instead of accumulating all the 
  outputs and compute them in the `test_epoch_end`.
  - Metrics are not implemented as Lightning Callbacks to avoid again returning and accumulating the values in the `eval_step`.
  - The confusion matrix is just a reminder of a good way to accumulate values over steps and to use already validated
  implementations like the ones from Scikit-Learn.

## Notes about PyTorch Lightning 
As mentioned before, in my experience I have found some default behaviors and good practices not clearly specified in
the PyTorch Lightning tutorials and documentation. Here is the list:
1. Return the minimum things possible in the `**_step()` functions because they will be accumulated for the `**_epoch_end()` function.
2. For the same reason, `.detach()` and move to cpu everything you need to return in `**_step()`, if possible.
3. In order to not mess up the logging and the multi-gpu training always use `self.log()` and not `self.logger.*`.
4. If you need to send a tensor to a device (like cuda:0) do it int [this](https://pytorch-lightning.readthedocs.io/en/stable/accelerators/accelerator_prepare.html)
way to not mess up the multi-gpu and tpu training.
5. By default PyTorch Lightning logs your experiment values and checkpoints. I would recommend to, instead of leaving
them by default, you modify the settings implemented in this template.
