from azureml.core import Environment
from azureml.core.model import InferenceConfig
from azureml.core import Workspace
from azureml.core import Model
from azureml.core.webservice import LocalWebservice
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.webservice import AciWebservice

deployment_config = AciWebservice.deploy_configuration(cpu_cores=0.1, 
                                               memory_gb=0.5, 
                                               tags={"data": "Hearth",  "method" : "sklearn"}, 
                                               description='Predict hearth attack with sklearn')

# Despliegue local mediante docker
#deployment_config = LocalWebservice.deploy_configuration(port=6789)

ws = Workspace.from_config()
model = Model(ws, 'heart_attack_model')
                   
print(model.name, model.id, model.version, sep='\t')

# Creciación de azure constainer con ambiente gestionado mediante conda.
myenv = Environment(name="myenv")
# Se habilita docker
myenv.docker.enabled = True
# Definición de dependencias de docker.
myenv.python.conda_dependencies = CondaDependencies.create(conda_packages=['scikit-learn'],
                                                           pip_packages=['azureml-defaults', 'numpy', 'pandas'])
                                                          
inf_config = InferenceConfig(environment=myenv, source_directory='./src', entry_script='entry.py')
service = Model.deploy(ws, "heart-webservice", [model], inf_config, deployment_config)
service.wait_for_deployment(show_output=True)
print(service.get_logs())