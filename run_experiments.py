from pathlib import Path
import json
import subprocess
import argparse
import mlflow
import os
from utils import pretty_dataset_name
import lib_jobs

os.environ["MLFLOW_TRACKING_USERNAME"] = "exp-01.mlflow-yang.tifreaa"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "parola"
remote_server_uri = "https://exp-01.mlflow-yang.inf.ethz.ch"
mlflow.set_tracking_uri(remote_server_uri)


def main():
    parser = argparse.ArgumentParser("Run experiments")
    parser.add_argument("--settings_json", type=str, required=True)
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument(
        "--py_to_run",
        type=str,
        required=True,
        help="""
Python program to run for each experiment. It must accept the following flags:
    --experiment_name (for mlflow logging)
    --id_dataset (to be used with lib_comparison.load_dataset)
    --ood_dataset (same as id_dataset)
""",
    )
    parser.add_argument("--local", action="store_true")
    args = parser.parse_args()

    settings = json.loads(Path(args.settings_json).read_text())

    for dataset_type, scenarios in settings.items():
        for scenario in scenarios:
            assert len(scenario) == 2
            id_dataset, ood_dataset = scenario

            if args.experiment_name is not None:
                curr_experiment_name = args.experiment_name
            else:
                curr_experiment_name = "{}_vs_{}".format(
                    pretty_dataset_name(id_dataset), pretty_dataset_name(ood_dataset)
                )

            # This call creates the experiment if it does not exist yet. It is necessary to avoid race conditions if
            # all the jobs want to create the same experiment.
            mlflow.set_experiment(curr_experiment_name)

            start_lr_values = {
                "adam": [0.001, 0.01, 0.1],
                "sgd": [0.01, 0.1]
            }
            for optimizer in ["sgd", "adam"]:
                for start_lr in start_lr_values:
                    for nnpu_threshold in [-0.1, 0.01, 0., 0.01, 0.1]:
                        for transductive in [True, False]:
                            cli_args = [
                                ("experiment_name", curr_experiment_name),
                                ("id_dataset", id_dataset),
                                ("ood_dataset", ood_dataset),
                                ("with_param", f"optimizer={optimizer}"),
                                ("with_param", f"learning_rate_cls={start_lr}"),
                                ("with_param", f"nn_threshold={nnpu_threshold}"),
                                ("with_param", f"transductive={transductive}"),
                            ]
                            gin_args = []

                            if args.local:
                                lib_jobs.launch_local(args.py_to_run, cli_args, gin_args)
                                continue

                        if "mnist" in id_dataset:
                            nhours = 1
                        elif "cifar10:" in id_dataset or "cifar100:" in id_dataset:
                            nhours = 4
                        elif "svhn_cropped:" in id_dataset:
                            nhours = 4
#                             nhours = 6
                        elif "imagenet" in id_dataset:
                            nhours = 24
                        else:
                            nhours = 6

                            lib_jobs.launch_bsub(
                                nhours,
                                args.py_to_run,
                                cli_args=cli_args,
                                gin_args=gin_args,
                            )


if __name__ == "__main__":
    main()