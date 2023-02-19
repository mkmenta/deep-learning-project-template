# Deep Learning project template
![Formatting and tests workflow](https://github.com/mkmenta/deep-learning-project-template/actions/workflows/main-action.yml/badge.svg)

Deep Learning base project template using the following stack
- [PyTorch](https://pytorch.org/)
- [PyTorch Lightning](https://www.pytorchlightning.ai/)
- [Neptune.ai](https://neptune.ai/)

## Environment setup
Install requirements:
```
python3 -m pip install -r requirements.txt
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