import lib_data
import os
import subprocess
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--id_dataset", type=str, required=True)
    parser.add_argument("--ood_dataset", type=str, required=True)
    parser.add_argument("--goal_tag", type=str)
    parser.add_argument(
        "-wp",
        "--with_param",
        type=str,
        action="append",
        default=[],
        help="Optional repeated argument of the form k=[v], "
             "will be included in the cartesian product of parameters, "
             "using k as the gin parameter name. "
             "Example usage: --with_param data.batch_size=[32,64,128]",
    )
    args = parser.parse_args()

    lib_data.setup_mlflow()
    os.environ["MLFLOW_EXPERIMENT_NAME"] = args.experiment_name

    params_path = ""
    if "mnist" in args.id_dataset:
        params_path = "params/mnist/my_nnPU.yml"
    else:
        params_path = "params/cifar10/nnPU.yml"

    cmd = [
        "python3",
        "pu_biased_n.py",
        "--id_dataset",
        args.id_dataset,
        "--ood_dataset",
        args.ood_dataset,
        "--experiment_name",
        args.experiment_name,
        "--goal_tag",
        args.goal_tag,
        "--params-path",
        params_path,
    ]
    flatten = lambda l: [x for sublist in l for x in sublist]
    cmd += flatten([["-wp", wp] for wp in args.with_param])

    subprocess.check_call(cmd)


if __name__ == "__main__":
    main()
