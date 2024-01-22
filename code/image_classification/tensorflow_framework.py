import tensorflow as tf
import tensorflow_datasets as tfds

import csv, datetime, math, random
import numpy as np
from sklearn.metrics import accuracy_score


class Tensorflow:
    def __init__(self):
        self.script_version = "1.0.1"
        self.version = tf.version.VERSION
        self.epochs = 0
        self.end_to_end_epochs = 0
        self.learning_rate = 0.0
        self.end_to_end_learning_rate = 0.0
        self.lr_scheduler = False
        self.transfer_learning = False

        self.train_steps_per_epoch = 0

        self.save_epoch_logs = False
        self.save_tensorboard_logs = False

    def set_random_seed(self, seed_val):
        """
        ## Configure Tensorflow for fixed seed runs
        """
        major, minor, revision = tf.version.VERSION.split(".")

        if int(major) >= 2 and int(minor) >= 7:
            # Sets all random seeds for the program (Python, NumPy, and TensorFlow).
            # Supported in TF 2.7.0+
            tf.keras.utils.set_random_seed(seed_val)
            print("Setting random seed using tf.keras.utils.set_random_seed()")
        else:
            # for TF < 2.7
            random.seed(seed_val)
            np.random.seed(seed_val)
            tf.random.set_seed(seed_val)
            print("Setting random seeds manually")

    def set_op_determinism(self):
        """
        ## Configure Tensorflow for deterministic operations
        """
        major, minor, revision = tf.version.VERSION.split(".")

        # Configures TensorFlow ops to run deterministically to enable reproducible
        # results with GPUs (Supported in TF 2.8.0+)
        if int(major) >= 2 and int(minor) >= 8:
            tf.config.experimental.enable_op_determinism()
            print("Enabled op determinism")
        else:
            print("Op determinism not supported")
            exit()

    def load_dataset(self, dataset_details, batch_size):
        dataset_name = dataset_details["name"]
        dataset_split = dataset_details["split"]
        dataset_shape = dataset_details["dataset_shape"]

        def preprocessing(image, label):
            image = tf.cast(image, tf.float32)
            if self.transfer_learning == False:
                image = tf.image.per_image_standardization(image)
            image = tf.image.resize(
                image, dataset_shape[:2], antialias=False, method="nearest"
            )
            return image, label

        def augmentation(image, label):
            image = tf.image.resize_with_crop_or_pad(
                image,
                int(dataset_shape[0] + (dataset_shape[0] * 0.2)),
                int(dataset_shape[1] + (dataset_shape[1] * 0.2)),
            )
            image = tf.image.random_crop(image, dataset_shape)
            image = tf.image.random_flip_left_right(image)
            return image, label

        train, val, test = tfds.load(
            dataset_name,
            split=dataset_split,
            as_supervised=True,
        )

        print(f"Number of training samples: {train.cardinality()}")
        print(f"Number of validation samples: {val.cardinality()}")
        print(f"Number of test samples: {test.cardinality()}")

        # Batch and prefetch the dataset
        train_dataset = (
            train.map(preprocessing)
            .map(augmentation)
            .shuffle(10000)
            .batch(batch_size, drop_remainder=True)
            .repeat()
            .prefetch(tf.data.AUTOTUNE)
        )
        val_dataset = (
            val.map(preprocessing).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        )
        test_dataset = (
            test.map(preprocessing).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        )

        # Calculate the number of steps per epoch so each epoch goes through the entire dataset once
        train_size = len(list(train))
        self.train_steps_per_epoch = train_size // batch_size

        return train_dataset, val_dataset, test_dataset

    def load_model(self, model_name, dataset_details):
        num_classes = dataset_details["num_classes"]
        dataset_shape = dataset_details["dataset_shape"]

        models = {
            "DenseNet121": {
                "application": tf.keras.applications.DenseNet121,
                "preprocess_input": tf.keras.applications.densenet.preprocess_input,
            },
            "DenseNet169": {
                "application": tf.keras.applications.DenseNet169,
                "preprocess_input": tf.keras.applications.densenet.preprocess_input,
            },
            "ResNet50V2": {
                "application": tf.keras.applications.resnet_v2.ResNet50V2,
                "preprocess_input": tf.keras.applications.resnet_v2.preprocess_input,
            },
            "ResNet101V2": {
                "application": tf.keras.applications.resnet_v2.ResNet101V2,
                "preprocess_input": tf.keras.applications.resnet_v2.preprocess_input,
            },
            "EfficientNetB0": {
                "application": tf.keras.applications.EfficientNetB0,
                "preprocess_input": tf.keras.applications.efficientnet.preprocess_input,
            },
            "EfficientNetB3": {
                "application": tf.keras.applications.EfficientNetB3,
                "preprocess_input": tf.keras.applications.efficientnet.preprocess_input,
            },
            "Xception": {
                "application": tf.keras.applications.Xception,
                "preprocess_input": tf.keras.applications.xception.preprocess_input,
            },
        }

        random_init_args = {
            "include_top": True,
            "weights": None,
            "input_shape": dataset_shape,
            "classes": num_classes,
            "classifier_activation": "softmax",
        }
        transfer_init_args = {
            "include_top": False,
            "weights": "imagenet",
            "input_shape": dataset_shape,
            "pooling": None,
        }

        if self.transfer_learning == False:
            base_model = models[model_name]["application"](**random_init_args)
            return base_model

        else:
            base_model = models[model_name]["application"](**transfer_init_args)

            # Freeze the base_model
            base_model.trainable = False

            inputs = tf.keras.Input(shape=dataset_shape)

            # Build the top of the model
            x = models[model_name]["preprocess_input"](inputs)
            x = base_model(x)

            x = tf.keras.layers.GlobalAveragePooling2D()(x)
            x = tf.keras.layers.Dropout(0.2)(x)

            outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

            transfer_model = tf.keras.Model(inputs, outputs)

            return transfer_model

    def train(
        self,
        model,
        train_dataset,
        val_dataset,
        epochs,
        csv_train_log_file=None,
        run_name="",
    ):
        """
        ## Define the learning rate schedule
        """

        def lr_schedule(epoch):
            if epoch < math.ceil(self.epochs * 0.5):
                return self.learning_rate
            elif epoch < math.ceil(self.epochs * 0.75):
                return self.learning_rate * 0.1
            else:
                return self.learning_rate * 0.01

        model.summary(show_trainable=True)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule(0)),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"],
        )

        # Define the learning rate scheduler callback
        lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_schedule)

        # Define csv logger callback
        csv_logger = tf.keras.callbacks.CSVLogger(csv_train_log_file)

        # Define tensorboard callback
        log_dir = (
            "logs/fit/"
            + run_name
            + "-"
            + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        )
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir, histogram_freq=1
        )

        # Define callbacks
        callbacks = []

        if self.save_epoch_logs:
            callbacks.append(csv_logger)

        if self.lr_scheduler:
            callbacks.append(lr_scheduler)

        if self.save_tensorboard_logs:
            callbacks.append(tensorboard_callback)

        # Train the model
        model.fit(
            train_dataset,
            epochs=epochs,
            steps_per_epoch=self.train_steps_per_epoch,
            validation_data=val_dataset,
            callbacks=callbacks,
        )

        if self.transfer_learning:
            # Unfreeze the base_model
            for layer in model.layers:
                if not isinstance(layer, tf.keras.layers.BatchNormalization):
                    layer.trainable = True

            model.summary(show_trainable=True)

            model.compile(
                optimizer=tf.keras.optimizers.Adam(
                    learning_rate=self.end_to_end_learning_rate
                ),  # Low learning rate
                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                metrics=["accuracy"],
            )

            # Define tensorboard callback
            log_dir = (
                "logs/fit/"
                + run_name
                + "-end-2-end-"
                + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            )
            tensorboard_callback = tf.keras.callbacks.TensorBoard(
                log_dir=log_dir, histogram_freq=1
            )

            # Define callbacks
            callbacks = []

            if self.save_tensorboard_logs:
                callbacks.append(tensorboard_callback)

            print("Fitting the end-to-end model")
            model.fit(
                train_dataset,
                epochs=self.end_to_end_epochs,
                steps_per_epoch=self.train_steps_per_epoch,
                validation_data=val_dataset,
                callbacks=callbacks,
            )

        return model

    def evaluate(
        self, model, test_dataset, save_predictions=False, predictions_csv_file=None
    ):
        # Get the predictions
        predictions = model.predict(test_dataset)

        # Get the labels of the validation dataset
        test_dataset = test_dataset.unbatch()
        labels = np.asarray(list(test_dataset.map(lambda x, y: y)))

        # Get the index to the highest probability
        y_true = labels
        y_pred = np.argmax(predictions, axis=1)

        if save_predictions:
            # Add the true values to the first column and the predicted values to the second column
            true_and_pred = np.vstack((y_true, y_pred)).T

            # Add each label predictions to the true and predicted values
            csv_output_array = np.concatenate((true_and_pred, predictions), axis=1)

            # Save the predictions to a csv file
            with open(predictions_csv_file, "w") as csvfile:
                writer = csv.writer(csvfile)

                csv_columns = ["true_value", "predicted_value"]
                for i in range(predictions.shape[1]):
                    csv_columns.append("label_" + str(i))

                writer.writerow(csv_columns)
                writer.writerows(csv_output_array.tolist())

        # Calucate the validation loss
        loss = tf.keras.losses.SparseCategoricalCrossentropy()
        validation_loss = loss(labels, predictions).numpy()

        # Use sklearn to calculate the validation accuracy
        validation_accuracy = accuracy_score(y_true, y_pred)

        return [validation_loss, validation_accuracy]

    def save(self, model, model_path):
        model.save(model_path)

    def load(self, model_path):
        model = tf.keras.models.load_model(model_path)

        return model
