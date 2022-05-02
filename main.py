import argparse
from datetime import datetime

import os
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam

import models
from constants import RESULTS_ROOT, NUM_MFCC, NO_features
from data_versions import DataVersions
from datastore import Datastore
from framework import train
from utils import str2dataset, get_datastore, combine_datastores, store_results, str2bool

time_str = datetime.now().strftime("%Y_%m_%d_%H_%M")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-version', nargs='+',
                        choices=[DataVersions.IEMOCAP, DataVersions.SAVEE, DataVersions.IMPROV, DataVersions.ESD,
                                 DataVersions.EMODB],
                        type=str2dataset, default=DataVersions.IEMOCAP)
    parser.add_argument('--data-split', nargs='+', type=float, default=None)
    parser.add_argument('--train-epochs', type=int, default=128)
    parser.add_argument('--loss', type=str, default='mse', choices=['mse', 'categorical_crossentropy', 'poisson'])
    parser.add_argument('--pre-train', type=str2bool, default=False)
    parser.add_argument('--pre-train-dataset',
                        choices=[DataVersions.IEMOCAP, DataVersions.IMPROV, DataVersions.SAVEE, DataVersions.ESD,
                                 DataVersions.EMODB],
                        type=str2dataset,
                        default=DataVersions.IEMOCAP)
    parser.add_argument('--pre-train-data-split', type=float, default=None)
    parser.add_argument('--pre-train-epochs', type=int, default=64)
    parser.add_argument('--testing-dataset', type=str2dataset, default=None,
                        choices=[DataVersions.IEMOCAP, DataVersions.IMPROV, DataVersions.SAVEE, DataVersions.ESD,
                                 DataVersions.EMODB,
                                 DataVersions.COMBINED])
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--save', type=str2bool, default=False)
    parser.add_argument('--model-load-file', type=str, required=False, default=None)
    parser.add_argument('--schedule-csv', type=str, default=None)
    parser.add_argument('--schedule-idx', type=int, default=None)

    args = parser.parse_args()

    for k in args.__dict__.keys():
        print("\t{} :\t{}".format(k, args.__dict__[k]))

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    print("Tensorflow version:", tf.__version__)

    batch_size = 128

    if os.path.exists(f'{RESULTS_ROOT}/{time_str}'):
        raise RuntimeError(f'Results directory {RESULTS_ROOT}/{time_str} is already exists')

    log_dir = f'{RESULTS_ROOT}/{time_str}/logs'

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    data_version_map = {}
    custom_data_split = []
    if args.data_split is not None:
        if len(args.data_split) == 1 and len(args.data_version) > 1:
            for i in range(len(args.data_version)):
                custom_data_split.append(args.data_split[0])
        elif 1 < len(args.data_split) != len(args.data_version) > 1:
            raise RuntimeError("--data-split either should have one value or similar to --data-version")
        else:
            custom_data_split = args.data_split
    else:
        for i in range(len(args.data_version)):
            custom_data_split.append(None)

    if len(args.data_version) == 1:
        target_datastore = get_datastore(data_version=args.data_version[0],
                                         custom_split=None if args.data_split is None else args.data_split[0])
        data_version_map[args.data_version[0]] = target_datastore
    else:
        ds = []
        for i in range(len(args.data_version)):
            d = get_datastore(data_version=args.data_version[i], custom_split=custom_data_split[i])
            data_version_map[args.data_version[i]] = d
            ds.append(d)
        target_datastore = combine_datastores(ds)

    input_layer = Input(shape=(1, NUM_MFCC, NO_features))

    model: Model = models.get_model_9_rl(input_layer, model_name_prefix='mfcc')
    model.compile(optimizer=Adam(learning_rate=.00025), metrics=['mae', 'accuracy'], loss=args.loss)

    if args.model_load_file is not None:
        model: Model = tf.keras.models.load_model(args.model_load_file)

    pre_train_datastore: Datastore = None
    if args.pre_train:
        if args.pre_train_dataset == args.data_version:
            raise RuntimeError("Pre-Train and Target datasets cannot be the same")
        else:
            pre_train_datastore = get_datastore(data_version=args.pre_train_dataset,
                                                custom_split=args.pre_train_data_split)

        assert pre_train_datastore is not None

        (x_train, y_train, y_gen_train), _ = pre_train_datastore.get_data()

        pre_train_log_dir = f'{log_dir}/pre_train'
        if not os.path.exists(pre_train_log_dir):
            os.makedirs(pre_train_log_dir)

        history, pre_trained_model = train(x=x_train.reshape((len(x_train), 1, NUM_MFCC, NO_features)),
                                           y=y_train,
                                           model=model,
                                           epochs=args.pre_train_epochs,
                                           log_base_dir=pre_train_log_dir,
                                           batch_size=batch_size)
        model = pre_trained_model

    # Training
    train_log_dir = f'{log_dir}/train'
    if not os.path.exists(train_log_dir):
        os.makedirs(train_log_dir)

    (x_train, y_train, y_gen_train), _ = target_datastore.get_data()
    history, model = train(x=x_train.reshape((len(x_train), 1, NUM_MFCC, NO_features)),
                           y=y_train,
                           model=model,
                           epochs=args.train_epochs,
                           log_base_dir=train_log_dir,
                           batch_size=batch_size)

    # Testing with Labelled Data
    testing_dataset = args.testing_dataset
    if testing_dataset is not None:
        if testing_dataset == DataVersions.COMBINED:
            if pre_train_datastore is not None:
                testing_datastore = combine_datastores([target_datastore, pre_train_datastore])
            else:
                testing_datastore = target_datastore
        else:
            testing_datastore = data_version_map[testing_dataset]
    else:
        # testing dataset is not defined
        if pre_train_datastore is not None:
            testing_datastore = combine_datastores([target_datastore, pre_train_datastore])
        else:
            testing_datastore = target_datastore

    x_test, y_test, _ = testing_datastore.get_testing_data()
    test_loss, test_mae, test_acc = model.evaluate(x_test.reshape((len(x_test), 1, NUM_MFCC, NO_features)),
                                                   y_test, verbose=1)

    print(f"Test\n\t Accuracy: {test_acc}")
    store_results(f"{log_dir}/results.txt", args=args, experiment="", time_str=time_str,
                  test_loss=test_loss, test_acc=test_acc)

    if args.save:
        model_dir = f'{RESULTS_ROOT}/{time_str}/model'
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        model.save(f"{model_dir}/{model.name}.h5")

    if args.schedule_csv is not None:
        from scheduler_callback import callback
        callback(args.schedule_csv, args.schedule_idx)


if __name__ == "__main__":
    main()
