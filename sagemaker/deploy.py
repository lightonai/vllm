import json
import os
import random
import re
import string

import boto3
import fire
import requests
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest


def generate_random_string(length):
    characters = string.ascii_letters + string.digits
    return "".join(random.choice(characters) for _ in range(length))


def get_sagemaker_vars(region, base_image_name):
    account_id = boto3.client("sts").get_caller_identity()["Account"]
    return (
        f"{account_id}.dkr.ecr.{region}.amazonaws.com/{base_image_name}",
        f"arn:aws:iam::{account_id}:role/SMRole",
    )


def read_json_file(config_path: str):
    with open(config_path, "r") as file:
        data = json.load(file)
    return data


def push_lora(*, lora_name, s3_uri, endpoint_name, region):
    data = {
        "endpoint": "/loras",
        "payload": {
            "lora_name": lora_name,
            "s3_uri": s3_uri,
        },
    }

    def sign_request(url, method="POST", region=region, data=None):
        session = boto3.Session()
        credentials = session.get_credentials()

        request = AWSRequest(method=method, url=url, data=json.dumps(data))
        SigV4Auth(credentials, "sagemaker", region).add_auth(request)

        return dict(request.headers)

    url = f"https://runtime.sagemaker.{region}.amazonaws.com/endpoints/{endpoint_name}/invocations"

    headers_auth = sign_request(url, data=data)
    headers = {
        "Content-Type": "application/json",
    }

    response = requests.post(url,
                             headers={
                                 **headers,
                                 **headers_auth
                             },
                             json=data)
    if response.status_code == 200:
        print(f"Successfully pushed LoRA {lora_name}")


def deploy(config_path: str):
    """
    Deploy a VLLM model to SageMaker.
    """

    config_data = read_json_file(config_path)
    model = config_data.get("model")
    served_model_name = config_data.get("served_model_name")
    image = config_data.get("image")
    instance_type = config_data.get("sagemaker_instance_type")
    loras = config_data.get("loras")
    region = config_data.get("region", "us-west-2")
    env_vars = config_data.get("env_vars", {})

    image_version = image.split(":")[-1].replace(".", "-")

    random_id = generate_random_string(5)
    name = model.split("/")[-1]
    endpoint_name = ("vllm-" + image_version + "--" +
                     re.sub("[^0-9a-zA-Z]", "-", name) + "-" + random_id)
    model_name = f"{endpoint_name}-mdl"
    endpoint_config_name = f"{endpoint_name}-epc"

    assert (len(endpoint_name) <=
            63), "Endpoint name must be less than 63 characters"
    assert len(model_name) <= 63, "Model name must be less than 63 characters"
    assert (len(endpoint_config_name) <=
            63), "Endpoint config name must be less than 63 characters"
    assert os.getenv("HF_TOKEN") is not None, "HF_TOKEN is required"

    assert model is not None

    for key, value in env_vars.items():
        assert isinstance(
            value, str
        ), f"env_vars values must be strings. Found '{key}' associated with '{value}' with type {type(value)}."

    if loras and len(loras) > 0:
        assert (env_vars.get("ENABLE_LORA") is not None
                ), "ENABLE_LORA env var is required when 'loras' are provided"
        assert (env_vars.get("MAX_LORAS") is not None
                ), "MAX_LORAS env var is required when 'loras' are provided"
        assert (
            env_vars.get("MAX_LORA_RANK") is not None
        ), "MAX_LORA_RANK env var is required when 'loras' are provided"
        for lora in loras:
            assert lora.get("lora_name") is not None, "lora_name is required"
            assert lora.get("s3_uri") is not None, "s3_uri is required"

    container_env = {
        "MODEL": model,
        "SERVED_MODEL_NAME": served_model_name or endpoint_name,
        "HF_TOKEN": os.getenv("HF_TOKEN"),
        **env_vars,
    }

    vllm_image_uri, role = get_sagemaker_vars(region, image)

    print("\nThis configuration will be applied: ")
    print(
        json.dumps(
            {
                "container_env": container_env,
                "instance_type": instance_type,
                "sagemaker_endpoint": endpoint_name,
                "sagemaker_model": model_name,
                "sagemaker_endpoint_config": endpoint_config_name,
                "region": region,
                "image_uri": vllm_image_uri,
            },
            indent=4,
        ))

    primary_container = {
        "Image": vllm_image_uri,
        "Environment": container_env,
    }

    # Ask for confirmation
    print("\nDo you want to continue? (yes/no)")
    response = input()
    if response != "yes":
        print("Exiting...")
        return

    # create model
    sm_client = boto3.client(service_name="sagemaker", region_name=region)
    create_model_response = sm_client.create_model(
        ModelName=model_name,
        ExecutionRoleArn=role,
        PrimaryContainer=primary_container,
    )
    print("Model Arn: " + create_model_response["ModelArn"])

    # create endpoint configuration
    create_endpoint_config_response = sm_client.create_endpoint_config(
        EndpointConfigName=endpoint_config_name,
        ProductionVariants=[{
            "InstanceType": instance_type,
            "InitialVariantWeight": 1,
            "InitialInstanceCount": 1,
            "ModelName": model_name,
            "VariantName": "AllTraffic",
        }],
    )
    print("Endpoint Config Arn: " +
          create_endpoint_config_response["EndpointConfigArn"])

    # create endpoint
    create_endpoint_response = sm_client.create_endpoint(
        EndpointName=endpoint_name,
        EndpointConfigName=endpoint_config_name,
    )

    print(f"Waiting for {endpoint_name} endpoint to be in service...")
    waiter = sm_client.get_waiter("endpoint_in_service")
    waiter.wait(EndpointName=endpoint_name)

    resp = sm_client.describe_endpoint(EndpointName=endpoint_name)
    print("Endpoint Arn: " + create_endpoint_response["EndpointArn"])
    print("Endpoint Status: " + resp["EndpointStatus"])
    print("=" * 20)
    print("Endpoint name: " + endpoint_name)
    print("=" * 20)

    if loras and len(loras) > 0:
        for lora in loras:
            push_lora(
                lora_name=lora.get("lora_name"),
                s3_uri=lora.get("s3_uri"),
                endpoint_name=endpoint_name,
                region=region,
            )

    return endpoint_name


if __name__ == "__main__":
    fire.Fire(deploy)
