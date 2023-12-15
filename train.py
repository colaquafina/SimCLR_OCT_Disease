import argparse
import importlib
import json
import os
from typing import Dict
import pdb

DEFAULT_TRAIN_ARGS = {"batch_size": 32, "epochs": 10, "lr": 1e-3, "loss": "crossentropy", "optimizer": "adam"}

def run_experiment(experiment_config: Dict, save_weights: bool):
    experiment_config["train_args"] = {
        **DEFAULT_TRAIN_ARGS,
        **experiment_config.get("train_args", {})
    }
    # experiment_config["gpu_ind"] = gpu_ind

    datasets_module = importlib.import_module("dataset")
    dataset_class_ = getattr(datasets_module, experiment_config["RetinaDataset"])
    dataset = dataset_class_()
    dataset.load()
    print (dataset)

    models_module = importlib.import_module("base_model")
    model_class_ = getattr(models_module, experiment_config["model"])

    networks_module = importlib.import_module("restina_model")
    network_fn_ = getattr(networks_module, experiment_config["network"])
    
    model = model_class_(
        dataset_cls=dataset_class_, network_fn=network_fn_
    )
    print (model)

    score = model.evaluate(dataset.test, batch_size=experiment_config["train_args"]["batch_size"])
    print(f"Test evaluation: {score}")
    
    if save_weights:
        model.save_weights()

def _parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--gpu", type=int, default=0, help="Provide index of GPU to use.")
    parser.add_argument(
        "--save",
        default=False,
        dest='save',
        action='store_true',
        help="If true, then final weights will be saved to canonical, version-controlled location",
    )
    parser.add_argument(
        "experiment_config",
        type=str,
        help='Experiment JSON (\'{"dataset": "RetinaDataset", "model": "RetinaModel", "network": "resnetconv"}\'',
    )
    parser.add_argument(
        "--nowandb", default=False, action="store_true", help="if true, do not use wandb"
    )
    args = parser.parse_args()
    return args

def main():
    args = _parse_args()

    experiment_config = json.loads(args.experiment_config)
    run_experiment(experiment_config, args.save)

if __name__ == "__main__":
    main()
