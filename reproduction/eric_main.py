import lib_data
import os
import subprocess
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--id_dataset", type=str, required=True)
    parser.add_argument("--ood_dataset", type=str, required=True)
    args = parser.parse_args()

    lib_data.setup_mlflow()
    os.environ["MLFLOW_EXPERIMENT_NAME"] = args.experiment_name

    params_path = ""
    if "mnist" in args.id_dataset:
        params_path = "params/mnist/my_nnPU.yml"

    for start_lr in [0.0001, 0.001, 0.01]:
        for nnpu_threshold in [-0.1, 0.01, 0., 0.01, 0.1]:
            subprocess.check_call(
                [
                    "python3",
                    "pu_biased_n.py",
                    "--id_dataset",
                    args.id_dataset,
                    "--ood_dataset",
                    args.ood_dataset,
                    "--experiment_name",
                    args.experiment_name,
                    "--params-path",
                    params_path,
                    "-wp",
                    f"learning_rate_cls={start_lr}",
                    "-wp",
                    f"nn_threshold={nnpu_threshold}",
                ]
            )


if __name__ == "__main__":
    main()
