from dflow import config, s3_config

def dflow_config(
        config_data,
):
    if config_data is not None:
        config["host"] = config_data.get('host', None)
        s3_config["endpoint"] = config_data.get('s3_endpoint', None)
        config["k8s_api_server"] = config_data.get('k8s_api_server', None)
        config["token"] = config_data.get('token', None)    
        
