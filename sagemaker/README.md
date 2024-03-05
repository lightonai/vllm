# Deploy a model to SageMaker

## Install dependencies

```bash
pip install -r requirements-deploy.txt
```

## Deploy a model

Deploy a model to SageMaker:

```bash
python sagemaker/deploy.py --config_path sagemaker/configs/mistral.json
```

> You can create more models configs in the `sagemaker/configs` folder.

To clean up the SageMaker resources:

```bash
python sagemaker/cleanup.py --endpoint_name <endpoint_name>
```
