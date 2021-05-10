from azureml.core import Workspace
from azureml.core import Experiment
from azureml.core import Environment
from azureml.core import ScriptRunConfig
from azureml.core import Model
import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser()      
    parser.add_argument(
        '--solver',
        type=str,
        default="liblinear",
        help='Solver para la regresión logistica'
    )
    parser.add_argument(
        '--random_state',
        type=int,
        default=42,
        help='Entero aleatorio'
    )
    args = parser.parse_args()

    ws = Workspace.from_config()
    experiment = Experiment(workspace=ws, name='cloud-heart-attack')

    config = ScriptRunConfig(
        source_directory='./src',
        script='remote-train.py',
        compute_target='cpu-cluster',
        arguments=['--solver', args.solver, '--random_state', args.random_state]
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



    