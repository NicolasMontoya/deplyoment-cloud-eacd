from azureml.core import Workspace
from azureml.core import Experiment
from azureml.core import Environment
from azureml.core import ScriptRunConfig
from azureml.core import Dataset
from azureml.core import Model


if __name__ == "__main__":
    ws = Workspace.from_config()
    experiment = Experiment(workspace=ws, name='cloud-heart-attack')

    config = ScriptRunConfig(
        source_directory='./src',
        script='remote-train.py',
        compute_target='cpu-cluster'
    )
    # set up pytorch environment
    env = Environment.from_conda_specification(
        name='sklearn-remote-env',
        file_path='./azure-config/compute-env-config.yml'
    )
    config.run_config.environment = env

    run = experiment.submit(config)
    aml_url = run.get_portal_url()
    print("Submitted to compute cluster. Click link below")
    print("")
    print(aml_url)
    run.wait_for_completion(show_output=True)
    run.register_model( model_name='heart_attack_model',
                    model_path='outputs/LogisticRegression.pkl', # run outputs path
                    description='A classification model LogisticRegression',
                    tags={'data-format': 'CSV'},
                    model_framework=Model.Framework.SCIKITLEARN,
                    model_framework_version='0.20.3')



    