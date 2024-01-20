import argparse, csv, os, sys, yaml
from datetime import datetime

script_version = "1.0.2"


def get_dataset_details(dataset_name):
    """
    ## Datasets definition dictionary
    """
    datasets = {
        "cifar100": {
            "name": "cifar100",
            "split": ["train[:80%]", "train[80%:100%]", "test"],
            "num_classes": 100,
            "dataset_shape": (128, 128, 3),
        },
        "cifar10": {
            "name": "cifar10",
            "split": ["train[:80%]", "train[80%:100%]", "test"],
            "num_classes": 10,
            "dataset_shape": (128, 128, 3),
        },
        "imagenette": {
            "name": "imagenette/320px-v2",
            "split": ["train[:80%]", "train[80%:100%]", "validation"],
            "num_classes": 10,
            "dataset_shape": (224, 224, 3),
        },
        "oxford_iiit_pet": {
            "name": "oxford_iiit_pet",
            "split": ["train[:80%]", "train[80%:100%]", "test"],
            "num_classes": 37,
            "dataset_shape": (299, 299, 3),
        },
        "oxford_flowers102": {
            "name": "oxford_flowers102",
            "split": ["train", "validation", "test"],
            "num_classes": 102,
            "dataset_shape": (299, 299, 3),
        },
        "stanford_dogs": {
            "name": "stanford_dogs",
            "split": ["train[:80%]", "train[80%:100%]", "test"],
            "num_classes": 120,
            "dataset_shape": (299, 299, 3),
        },
    }

    return datasets[dataset_name]


def image_classification(
    machine_learning_framework="TensorFlow",
    model_name="ResNet20",
    dataset_name="cifar10",
    op_determinism=False,
    batch_size=128,
    learning_rate=0.001,
    end_to_end_learning_rate=0.0,
    lr_scheduler=False,
    epochs=200,
    end_to_end_epochs=0,
    seed_val=1,
    transfer_learning=False,
    run_name="",
    start="",
    save_model=False,
    save_predictions=False,
    save_epoch_logs=False,
    save_tensorboard_logs=False,
):
    if machine_learning_framework == "TensorFlow":
        from tensorflow_framework import Tensorflow

        framework = Tensorflow()
        framework.epochs = epochs
        framework.end_to_end_epochs = end_to_end_epochs
        framework.learning_rate = learning_rate
        framework.end_to_end_learning_rate = end_to_end_learning_rate
        framework.lr_scheduler = lr_scheduler
        framework.transfer_learning = transfer_learning

        framework.save_epoch_logs = save_epoch_logs
        framework.save_tensorboard_logs = save_tensorboard_logs

    if seed_val != 1:
        """
        ## Configure framework for fixed seed runs
        """
        framework.set_random_seed(seed_val)

    if op_determinism:
        """
        ## Configure framework deterministic operations
        """
        framework.set_op_determinism()

    """
    ## Get the dataset details
    """
    dataset_details = get_dataset_details(dataset_name)

    """
    ## Load the dataset
    """
    # Always use the same random seed for the dataset
    train_dataset, val_dataset, test_dataset = framework.load_dataset(
        dataset_details, batch_size
    )

    """
    ## Create the model
    """
    model = framework.load_model(model_name, dataset_details)

    """
    ## Create the base name for the log and model files
    """
    base_name = os.path.basename(sys.argv[0]).split(".")[0]

    if len(run_name) >= 1:
        base_name = run_name

    """
    ## Train the model
    """
    # Time the training
    start_time = datetime.now()

    if seed_val != 1:
        csv_train_log_file = "%s_log_%s_%s.csv" % (
            base_name,
            machine_learning_framework,
            seed_val,
        )
    else:
        csv_train_log_file = "%s_log_%s_%s.csv" % (
            base_name,
            machine_learning_framework,
            start,
        )

    # Train the model
    trained_model = framework.train(
        model,
        train_dataset,
        val_dataset,
        epochs,
        csv_train_log_file,
        run_name,
    )

    # Calculate the training time
    end_time = datetime.now()
    training_time = end_time - start_time

    """
    ## Evaluate the trained model and save the predictions
    """
    prediction_path = ""

    if save_predictions:
        if os.path.exists("predictions/") == False:
            os.mkdir("predictions/")

        prediction_path = "predictions/"

        if run_name != "":
            if os.path.exists("predictions/" + run_name + "/") == False:
                os.mkdir("predictions/" + run_name + "/")
            prediction_path = "predictions/" + run_name + "/"

    if seed_val != 1:
        predictions_csv_file = (
            prediction_path + base_name + "_seed_" + str(seed_val) + ".csv"
        )
    else:
        predictions_csv_file = (
            prediction_path
            + base_name
            + "_ts_"
            + start
            + ".csv"
        )

    score = framework.evaluate(
        trained_model, test_dataset, save_predictions, predictions_csv_file
    )

    print("Test loss: ", score[0])
    print("Test accuracy: ", score[1])
    print("Training time: ", training_time)

    """
    ## Save the model
    """
    if save_model:
        if os.path.exists("models/") == False:
            os.mkdir("models/")

        model_path = "models/"

        if run_name != "":
            if os.path.exists("models/" + run_name + "/") == False:
                os.mkdir("models/" + run_name + "/")
            model_path = "models/" + run_name + "/"

        if seed_val != 1:
            model_path = model_path + base_name + "_seed_" + str(seed_val)
        else:
            model_path = (
                model_path
                + base_name
                + "_ts_"
                + start
            )

        # Append the file extension based on the machine learning framework
        if machine_learning_framework == "TensorFlow":
            model_path = model_path + ".keras"

        framework.save(trained_model, model_path)

    return score[0], score[1], training_time


def get_system_info(filename):
    if os.path.exists("system_info.py"):
        import system_info

        sysinfo = system_info.get_system_info()

        with open("%s.yaml" % filename, "w") as system_info_file:
            yaml.dump(sysinfo, system_info_file, default_flow_style=False)

        return sysinfo
    else:
        return None


def save_score(
    test_loss,
    test_accuracy,
    machine_learning_framework,
    batch_size,
    learning_rate,
    end_to_end_learning_rate,
    lr_scheduler,
    epochs,
    end_to_end_epochs,
    training_time,
    model_name,
    dataset_name,
    op_determinism,
    seed_val,
    transfer_learning,
    filename,
    run_name="",
    start="",
):
    if machine_learning_framework == "TensorFlow":
        from tensorflow_framework import Tensorflow

        framework = Tensorflow()

    csv_file = filename + ".csv"
    write_header = False

    # If determistic is false and the seed value is 1 then the
    # seed value is totally random and we don't know what it is.
    if seed_val == 1:
        seed_val = "random"

    if not os.path.isfile(csv_file):
        write_header = True

    with open(csv_file, "a") as csvfile:
        fieldnames = [
            "run_name",
            "script_version",
            "date_time",
            "fit_time",
            "python_version",
            "machine_learning_framework",
            "framework_version",
            "batch_size",
            "learning_rate",
            "end_to_end_learning_rate",
            "lr_scheduler",
            "epochs",
            "end_to_end_epochs",
            "model_name",
            "dataset_name",
            "random_seed",
            "op_determinism",
            "transfer_learning",
            "test_loss",
            "test_accuracy",
        ]

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if write_header:
            writer.writeheader()

        writer.writerow(
            {
                "run_name": run_name,
                "script_version": script_version,
                "date_time": start,
                "fit_time": int(training_time.total_seconds()),
                "python_version": sys.version.replace("\n", ""),
                "machine_learning_framework": machine_learning_framework,
                "framework_version": framework.version,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "end_to_end_learning_rate": end_to_end_learning_rate,
                "lr_scheduler": lr_scheduler,
                "epochs": epochs,
                "end_to_end_epochs": end_to_end_epochs,
                "model_name": model_name,
                "dataset_name": dataset_name,
                "random_seed": seed_val,
                "op_determinism": op_determinism,
                "transfer_learning": transfer_learning,
                "test_loss": test_loss,
                "test_accuracy": test_accuracy,
            }
        )


def parse_arguments(args):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--op-determinism",
        dest="op_determinism",
        help="Run with deterministic operations",
        action="store_true",
    )

    parser.add_argument(
        "--seed-val", dest="seed_val", help="Set the seed value", type=int, default=1
    )

    parser.add_argument(
        "--batch-size",
        dest="batch_size",
        help="Size of the mini-batches",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--learning-rate",
        dest="learning_rate",
        help="Base learning rate",
        type=float,
        default=0.001,
    )

    parser.add_argument(
        "--end-to-end-learning-rate",
        dest="end_to_end_learning_rate",
        help="end to end training learning rate",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--lr-scheduler",
        dest="lr_scheduler",
        help="Use the learning rate scheduler",
        action="store_true",
    )

    parser.add_argument(
        "--epochs",
        dest="epochs",
        help="Number of epochs",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--end-to-end-epochs",
        dest="end_to_end_epochs",
        help="Number of epochs for end to end training",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--run-name",
        dest="run_name",
        help="Name of training run",
        default="",
    )

    parser.add_argument(
        "--save-filename",
        dest="save_filename",
        help="filename used to save the results",
        type=str,
        default=str(os.path.basename(sys.argv[0]).split(".")[0]),
    )

    parser.add_argument(
        "--save-model", dest="save_model", help="Save the model", action="store_true"
    )

    parser.add_argument(
        "--save-predictions",
        dest="save_predictions",
        help="Save the predictions",
        action="store_true",
    )

    parser.add_argument(
        "--save-epoch-logs",
        dest="save_epoch_logs",
        help="Save the accuracy and loss logs for each epoch",
        action="store_true",
    )

    parser.add_argument(
        "--tensorboard",
        dest="save_tensorboard_logs",
        help="Save TensorBoard logs",
        action="store_true",
    )

    parser.add_argument(
        "--ml-framework",
        dest="machine_learning_framework",
        help="Name of Machine Learning framework",
        default="TensorFlow",
        choices=[
            "TensorFlow",
        ],
        required=True,
    )

    parser.add_argument(
        "--model-name",
        dest="model_name",
        help="Name of model to train",
        default="Xception",
        choices=[
            "DenseNet121",
            "DenseNet169",
            "EfficientNetB0",
            "EfficientNetB3",
            "ResNet50V2",
            "ResNet101V2",
            "Xception",
        ],
        required=True,
    )

    parser.add_argument(
        "--dataset-name",
        dest="dataset_name",
        help="cifar10, cifar100, imagenette, oxford_flowers102, oxford_iiit_pet, stanford_dogs, dtd, caltech101",
        default="cifar10",
        choices=[
            "cifar10",
            "cifar100",
            "imagenette",
            "oxford_iiit_pet",
            "oxford_flowers102",
            "stanford_dogs",
            "dtd",
            "caltech101",
        ],
        required=True,
    )

    parser.add_argument(
        "--transfer-learning",
        dest="transfer_learning",
        help="Use transfer learning to train the model using the imagenet weights",
        action="store_true",
    )

    return parser.parse_args(args)


if __name__ == "__main__":
    args = parse_arguments(sys.argv[1:])

    save_filename = args.save_filename

    system_info = get_system_info(save_filename)
    seed_val = args.seed_val
    epochs = args.epochs

    start = datetime.now().strftime("%Y%m%d%H%M%S")

    print(
        "\nImage Classification (%s - %s - %s): [%s]\n======================\n"
        % (
            args.machine_learning_framework,
            args.model_name,
            args.dataset_name,
            seed_val,
        )
    )
    test_loss, test_accuracy, training_time = image_classification(
        machine_learning_framework=args.machine_learning_framework,
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        op_determinism=args.op_determinism,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        end_to_end_learning_rate=args.end_to_end_learning_rate,
        lr_scheduler=args.lr_scheduler,
        epochs=epochs,
        end_to_end_epochs=args.end_to_end_epochs,
        seed_val=seed_val,
        transfer_learning=args.transfer_learning,
        run_name=args.run_name,
        start=start,
        save_model=args.save_model,
        save_predictions=args.save_predictions,
        save_epoch_logs=args.save_epoch_logs,
        save_tensorboard_logs=args.save_tensorboard_logs,
    )
    save_score(
        test_loss=test_loss,
        test_accuracy=test_accuracy,
        machine_learning_framework=args.machine_learning_framework,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        end_to_end_learning_rate=args.end_to_end_learning_rate,
        lr_scheduler=args.lr_scheduler,
        epochs=epochs,
        end_to_end_epochs=args.end_to_end_epochs,
        training_time=training_time,
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        op_determinism=args.op_determinism,
        seed_val=seed_val,
        transfer_learning=args.transfer_learning,
        filename=save_filename,
        run_name=args.run_name,
        start=start,
    )
