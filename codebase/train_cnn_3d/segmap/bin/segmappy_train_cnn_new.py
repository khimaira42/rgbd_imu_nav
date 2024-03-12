#!/usr/bin/env python
from __future__ import print_function
import numpy as np
import sys
import os
from tqdm import tqdm
# import ensure_segmappy_is_installed
from segmappy.segmappy.core.config import Config 
from segmappy.segmappy.core.dataset import Dataset
from segmappy.segmappy.core.generator import Generator
from segmappy.segmappy.tools.classifiertools import get_default_dataset, get_default_preprocessor
from segmappy.segmappy.tools.roccurve import get_roc_pairs, get_roc_curve
from segmappy.segmappy.models.model_groups_tf import init_model
import tensorflow as tf
# read config file
configfile = "haoranDrone.ini"
config = Config(configfile)

# add command line arguments to config
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--log")
parser.add_argument("--debug", action="store_true")
parser.add_argument("--retrain")
parser.add_argument("--checkpoints", type=int, default=1)
parser.add_argument("--keep-best", action="store_true")
parser.add_argument("--roc", action="store_true")
args = parser.parse_args()
config.log_name = args.log
config.debug = args.debug
config.checkpoints = args.checkpoints
config.retrain = args.retrain
config.keep_best = args.keep_best
config.roc = args.roc

# create or empty the model folder
if not os.path.exists(config.cnn_model_folder):
    os.makedirs(config.cnn_model_folder)
else:
    import glob
    model_files = glob.glob(os.path.join(config.cnn_model_folder, "*"))
    for model_file in model_files:
        os.remove(model_file)

# load preprocessor
preprocessor = get_default_preprocessor(config)

segments = []
classes = np.array([], dtype=int)
n_classes = 0
duplicate_classes = np.array([], dtype=int)
max_duplicate_class = 0
duplicate_ids = np.array([], dtype=int)
positions = []

runs = config.cnn_train_folders.split(",")
for run in runs:
    dataset = get_default_dataset(config, run)

    run_segments, run_positions, run_classes, run_n_classes = dataset.load(
        preprocessor=preprocessor
    )
    run_duplicate_classes = dataset.duplicate_classes
    run_duplicate_ids = dataset.duplicate_ids

    run_classes += n_classes

    run_duplicate_classes += max_duplicate_class

    segments += run_segments
    positions.append(run_positions)
    classes = np.concatenate((classes, run_classes), axis=0)
    n_classes += run_n_classes
    duplicate_classes = np.concatenate(
        (duplicate_classes, run_duplicate_classes), axis=0
    )
    duplicate_ids = np.concatenate((duplicate_ids, run_duplicate_ids), axis=0)

    max_duplicate_class = np.max(duplicate_classes) + 1

if config.debug:
    import json

    # empty or create the debug folder
    if os.path.isdir(config.debug_path):
        import glob

        debug_files = glob.glob(os.path.join(config.debug_path, "*.json"))
        for debug_file in debug_files:
            os.remove(debug_file)
    else:
        os.makedirs(config.debug_path)

    # store loss information
    epoch_log = []
    train_loss_log = []
    train_loss_c_log = []
    train_loss_r_log = []
    train_accuracy_log = []
    test_loss_log = []
    test_loss_c_log = []
    test_loss_r_log = []
    test_accuracy_log = []

    # store segment centers for the current run
    centers = []
    for cls in range(n_classes):
        class_ids = np.where(classes == cls)[0]
        last_id = class_ids[np.argmax(duplicate_ids[class_ids])]
        centers.append(np.mean(segments[last_id], axis=0).tolist())

    with open(os.path.join(config.debug_path, "centers.json"), "w") as fp:
        json.dump(centers, fp)

    # info for sequence prediction visualization
    pred = [0] * (np.max(duplicate_classes) + 1)
    duplicate_ids_norm = np.zeros(duplicate_ids.shape, dtype=int)
    for duplicate_class in np.unique(duplicate_classes):
        segment_ids = np.where(duplicate_classes == duplicate_class)[0]
        pred[duplicate_class] = [None] * segment_ids.size

        for i, segment_id in enumerate(segment_ids):
            duplicate_ids_norm[segment_id] = i

    def debug_write_pred(segment_id, segment_probs, train):
        top5_classes = np.argsort(segment_probs)[::-1]
        top5_classes = top5_classes[:5]
        top5_prob = segment_probs[top5_classes]

        segment_class = classes[segment_id]
        segment_prob = segment_probs[segment_class]

        info = [
            int(train),
            int(segment_class),
            float(segment_prob),
            top5_classes.tolist(),
            top5_prob.tolist(),
        ]

        duplicate_class = duplicate_classes[segment_id]
        duplicate_id = duplicate_ids_norm[segment_id]

        pred[duplicate_class][duplicate_id] = info


# initialize preprocessor
preprocessor.init_segments(segments, classes, train_ids=n_classes, positions=positions)

# initialize segment batch generators
gen_train = Generator(
    preprocessor,
    classes,
    n_classes,
    train=True,
    batch_size=config.batch_size,
    shuffle=True,
)

# load dataset for calculating roc
if config.roc:
    # get test dataset
    roc_preprocessor = get_default_preprocessor(config)

    roc_dataset = Dataset(
        folder=config.cnn_roc_folder,
        base_dir=config.base_dir,
        keep_match_thresh=0.7,
        require_change=0.1,
        require_relevance=0.05,
    )

    roc_segments, roc_positions, roc_classes, roc_n_classes = roc_dataset.load(
        preprocessor=roc_preprocessor
    )

    roc_duplicate_classes = roc_dataset.duplicate_classes

    # get roc positive and negative pairs
    pair_ids, pair_labels = get_roc_pairs(
        roc_segments, roc_classes, roc_duplicate_classes
    )

    roc_ids = np.unique(pair_ids)
    roc_segments = [roc_segments[roc_id] for roc_id in roc_ids]
    roc_classes = roc_classes[roc_ids]
    roc_positions = roc_positions[roc_ids]

    for i, roc_id in enumerate(roc_ids):
        pair_ids[pair_ids == roc_id] = i

    roc_preprocessor.init_segments(
        roc_segments, roc_classes, positions=roc_positions
    )

    # roc generator
    gen_roc = Generator(
        roc_preprocessor,
        range(len(roc_segments)),
        roc_n_classes,
        train=False,
        batch_size=config.batch_size,
        shuffle=False,
    )



# get test dataset
test_preprocessor = get_default_preprocessor(config)
test_segments = []
test_classes = np.array([], dtype=int)
test_n_classes = 0
test_positions = []

runs2 = config.cnn_test_folder

dataset = get_default_dataset(config, runs2)

test_segments, test_positions, test_classes, test_n_classes = dataset.load(
    preprocessor=test_preprocessor
)
# run_classes2 += test_n_classes

# test_segments += run_segments2
# test_positions.append(run_positions2)
# test_classes = np.concatenate((test_classes, run_classes2), axis=0)
# test_n_classes += run_n_classes2

# test_dataset = Dataset(
#     folder=config.cnn_test_folder,
#     base_dir=config.base_dir,
#     keep_match_thresh=0.7,
#     require_change=0.1,
#     require_relevance=0.05,
# )
test_preprocessor.init_segments(test_segments, test_classes, train_ids=test_n_classes, positions=test_positions)

# test_segments, test_positions, test_classes, test_n_classes = test_dataset.load(
#     preprocessor=test_preprocessor
# )
# test_duplicate_classes = test_dataset.duplicate_classes

# # get roc positive and negative pairs
# pair_ids, pair_labels = get_roc_pairs(
#     test_segments, test_classes, test_duplicate_classes
# )

# test_ids = np.unique(pair_ids)
# test_segments = [test_segments[test_id] for test_id in test_ids]
# test_classes = test_classes[test_ids]
# test_positions = test_positions[test_ids]

# for i, test_id in enumerate(test_ids):
#     pair_ids[pair_ids == test_id] = i

# test_preprocessor.init_segments(
#     test_segments, test_classes, positions=test_positions
# )

# roc generator
gen_test = Generator(
    test_preprocessor,
    test_classes,
    test_n_classes,
    train=False,
    batch_size=config.batch_size,
    shuffle=False,
)
print("Training with %d segments" % gen_train.n_segments)
print("Testing with %d segments" % gen_test.n_segments)

import tensorflow as tf

tf.compat.v1.disable_eager_execution()

tf.compat.v1.reset_default_graph()


if config.retrain:
    # restore variable names from previous session
    saver = tf.compat.v1.train.import_meta_graph(config.retrain + ".meta")
else:
    # define a new model
    init_model(tuple(preprocessor.voxels), n_classes)

    # model saver
    saver = tf.compat.v1.train.Saver(max_to_keep=config.checkpoints)


# get key tensorflow variables
cnn_graph = tf.compat.v1.get_default_graph()

cnn_input = cnn_graph.get_tensor_by_name("InputScope/input:0")
y_true = cnn_graph.get_tensor_by_name("y_true:0")
training = cnn_graph.get_tensor_by_name("training:0")
scales = cnn_graph.get_tensor_by_name("scales:0")

loss = cnn_graph.get_tensor_by_name("loss:0")
loss_c = cnn_graph.get_tensor_by_name("loss_c:0")
loss_r = cnn_graph.get_tensor_by_name("loss_r:0")

accuracy = cnn_graph.get_tensor_by_name("accuracy:0")
y_prob = cnn_graph.get_tensor_by_name("y_prob:0")
descriptor = cnn_graph.get_tensor_by_name("OutputScope/descriptor_read:0")
roc_auc = cnn_graph.get_tensor_by_name("roc_auc:0")

#global_step = cnn_graph.get_tensor_by_name("global_step:0")
#update_step = cnn_graph.get_tensor_by_name("update_step:0")
#update_step = cnn_graph.get_operation_by_name("update_step")
train_op = cnn_graph.get_operation_by_name("train_op")

summary_batch = tf.compat.v1.summary.merge_all("summary_batch")
summary_epoch = tf.compat.v1.summary.merge_all("summary_epoch")

global_step = tf.Variable(0, trainable=False, name="global_step")
update_step = tf.compat.v1.assign(
        global_step, tf.add(global_step, tf.constant(1)), name="update_step"
    )
with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    # tensorboard statistics
    if config.log_name:
        train_writer = tf.compat.v1.summary.FileWriter(
            os.path.join(config.log_path, config.log_name, "train"), sess.graph
        )
        test_writer = tf.compat.v1.summary.FileWriter(
            os.path.join(config.log_path, config.log_name, "test")
        )

    # initialize all tf variables
    tf.compat.v1.global_variables_initializer().run()

    if config.retrain:
        saver.restore(sess, config.retrain)

    # remember best epoch accuracy
    if config.keep_best:
        best_accuracy = 0

    # sequence of train and test batches
    batches = np.array([1] * gen_train.n_batches + [0] * gen_test.n_batches)
    #print('batches',batches)
    for epoch in tqdm(range(0, config.n_epochs)):
        print('global step',global_step)
        initial_step_value = sess.run(global_step)
        # ... rest of your code for the epoch ...
        #print('Epoch ', epoch, ' starts!')
        train_loss = 0
        train_loss_c = 0
        train_loss_r = 0
        train_accuracy = 0
        train_step = 0

        test_loss = 0
        test_loss_c = 0
        test_loss_r = 0
        test_accuracy = 0
        test_step = 0

        np.random.shuffle(batches)

        console_output_size = 0
        for step, train in enumerate(batches):
            #print('Batch starts!')
            if train:
                

                batch_segments, batch_classes = gen_train.next()

                # run optimizer and calculate loss and accuracy
                summary, batch_loss, batch_loss_c, batch_loss_r, batch_accuracy, batch_prob, _ = sess.run(
                    [summary_batch, loss, loss_c, loss_r, accuracy, y_prob, train_op],
                    feed_dict={
                        cnn_input: batch_segments,
                        y_true: batch_classes,
                        training: True,
                        scales: preprocessor.last_scales,
                    },
                )

                if config.debug:
                    for i, segment_id in enumerate(gen_train.batch_ids):
                        debug_write_pred(segment_id, batch_prob[i], train)

                if config.log_name:
                    #step_value = sess.run(global_step)
                    #print("Step value:", step_value, "Type:", type(step_value))

                    train_writer.add_summary(summary, sess.run(global_step))

                train_loss += batch_loss
                train_loss_c += batch_loss_c
                train_loss_r += batch_loss_r
                train_accuracy += batch_accuracy
                train_step += 1
                # ... your code for the training step ...

            else:
                batch_segments, batch_classes = gen_train.next()

                # calculate test loss and accuracy
                summary, batch_loss, batch_loss_c, batch_loss_r, batch_accuracy, batch_prob = sess.run(
                    [summary_batch, loss, loss_c, loss_r, accuracy, y_prob],
                    feed_dict={
                        cnn_input: batch_segments,
                        y_true: batch_classes,
                        training: False,
                        scales: preprocessor.last_scales,
                    },
                )

                if config.debug:
                    for i, segment_id in enumerate(gen_train.batch_ids):
                        debug_write_pred(segment_id, batch_prob[i], train)

                if config.log_name:
                    print('global step before add summary',sess.run(global_step))
                    test_writer.add_summary(summary, sess.run(global_step))

                test_loss += batch_loss
                test_loss_c += batch_loss_c
                test_loss_r += batch_loss_r
                test_accuracy += batch_accuracy
                test_step += 1

                # update training step
                sess.run(update_step)
                step_value = sess.run(global_step)
                print("Current global step:", step_value)

            # print results
            sys.stdout.write("\b" * console_output_size)

            console_output = "epoch %2d train_step %2d test_step %2d " % (epoch, train_step, test_step)

            if train_step:
                console_output += "loss %.4f acc %.2f c %.4f r %.4f " % (
                    train_loss / train_step,
                    train_accuracy / train_step * 100,
                    train_loss_c / train_step,
                    train_loss_r / train_step,
                )

            if test_step:
                console_output += "v_loss %.4f v_acc %.2f v_c %.4f v_r %.4f" % (
                    test_loss / test_step,
                    test_accuracy / test_step * 100,
                    test_loss_c / test_step,
                    test_loss_r / test_step,
                )
            console_output_size = len(console_output)

            sys.stdout.write(console_output)
            sys.stdout.flush()

        # dump prediction values and loss
        if config.debug:
            epoch_debug_file = os.path.join(config.debug_path, "%d.json" % epoch)
            with open(epoch_debug_file, "w") as fp:
                json.dump(pred, fp)

            epoch_log.append(epoch)
            train_loss_log.append(train_loss / train_step)
            train_loss_c_log.append(train_loss_c / train_step)
            train_loss_r_log.append(train_loss_r / train_step)
            train_accuracy_log.append(train_accuracy / train_step * 100)
            test_loss_log.append(test_loss / test_step)
            test_loss_c_log.append(test_loss_c / test_step)
            test_loss_r_log.append(test_loss_r / test_step)
            test_accuracy_log.append(test_accuracy / test_step * 100)

            with open(os.path.join(config.debug_path, "loss.json"), "w") as fp:
                json.dump(
                    {
                        "epoch": epoch_log,
                        "train_loss": train_loss_log,
                        "train_loss_c": train_loss_c_log,
                        "train_loss_r": train_loss_r_log,
                        "train_accuracy": train_accuracy_log,
                        "test_loss": test_loss_log,
                        "test_loss_c": test_loss_c_log,
                        "test_loss_r": test_loss_r_log,
                        "test_accuracy": test_accuracy_log,
                    },
                    fp,
                )

        # calculate roc
        if config.roc:
            cnn_features = []
            for batch in range(gen_roc.n_batches):
                batch_segments, _ = gen_roc.next()

                batch_descriptors = sess.run(
                    descriptor,
                    feed_dict={
                        cnn_input: batch_segments,
                        scales: roc_preprocessor.last_scales,
                    },
                )

                for batch_descriptor in batch_descriptors:
                    cnn_features.append(batch_descriptor)

            cnn_features = np.array(cnn_features)

            _, _, epoch_roc_auc = get_roc_curve(cnn_features, pair_ids, pair_labels)

            summary = sess.run(summary_epoch, feed_dict={roc_auc: epoch_roc_auc})
            test_writer.add_summary(summary, sess.run(global_step))

            sys.stdout.write(" roc: %.3f" % epoch_roc_auc)

        # flush tensorboard log
        if config.log_name:
            train_writer.flush()
            test_writer.flush()

        # save epoch model
        if not config.keep_best or test_accuracy > best_accuracy:
            if config.checkpoints > 1:
                model_name = "model-%d.ckpt" % sess.run(global_step)
            else:
                model_name = "model.ckpt"

            saver.save(sess, os.path.join(config.cnn_model_folder, model_name))
            tf.io.write_graph(
                sess.graph.as_graph_def(), config.cnn_model_folder, "graph.pb"
            )
        # ... code that processes each batch ...

        # End of an epoch
        final_step_value = sess.run(global_step)
        print("Final step value at the end of the epoch:", final_step_value)

