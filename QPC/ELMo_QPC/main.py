import sys
sys.path.append('../..')
import os
import numpy as np
import ujson as json
import tensorflow as tf
from tqdm import tqdm

from model import QPCModel
from LIB.utils import get_batch_dataset, get_dataset, write_metrics, save
from utils import get_record_parser_qqp


def train(config):
    with open(config.train_eval_file, "r") as fh:
        train_eval_file = json.load(fh)
    with open(config.dev_eval_file, "r") as fh:
        dev_eval_file = json.load(fh)
    with open(config.dev_meta, "r") as fh:
        meta = json.load(fh)

    dev_total = meta["total"]
    print("Building model...")
    best_acc, best_ckpt = 0., 0
    parser = get_record_parser_qqp(config)
    graph = tf.Graph()
    with graph.as_default() as g:
        train_dataset = get_batch_dataset(config.train_record_file, parser, config.batch_size)
        dev_dataset = get_dataset(config.dev_record_file, parser, config.batch_size)
        train_iterator = train_dataset.make_one_shot_iterator()
        dev_iterator = dev_dataset.make_one_shot_iterator()

        model = QPCModel(config, graph=g)

        sess_config = tf.ConfigProto(allow_soft_placement=True)
        sess_config.gpu_options.allow_growth = True

        with tf.Session(config=sess_config) as sess:
            writer = tf.summary.FileWriter(config.output_dir)
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(max_to_keep=1000)
            if os.path.exists(os.path.join(config.output_dir, "checkpoint")):
                saver.restore(sess, tf.train.latest_checkpoint(config.output_dir))
            if os.path.exists(config.best_ckpt):
                with open(config.best_ckpt, "r") as fh:
                    best_qqp_ckpt = json.load(fh)
                    best_acc, best_ckpt = float(best_qqp_ckpt["best_acc"]), int(best_qqp_ckpt["best_ckpt"])

            global_step = max(sess.run(model.global_step), 1)
            train_next_element = train_iterator.get_next()
            for _ in tqdm(range(global_step, config.num_steps + 1)):
                global_step = sess.run(model.global_step) + 1
                que1, que2, label, qa_id = sess.run(train_next_element)
                loss, pred_label, _ = sess.run([model.loss, model.pred_label, model.train_op], feed_dict={
                    model.que1: que1, model.que2: que2, model.label: label,
                    model.dropout: config.dropout, model.qa_id: qa_id,
                })
                if global_step % config.period == 0:
                    loss_sum = tf.Summary(value=[tf.Summary.Value(
                            tag="model/loss", simple_value=loss), ])
                    writer.add_summary(loss_sum, global_step)
                if global_step % config.checkpoint == 0:
                    filename = os.path.join(
                            config.output_dir, "model_{}.ckpt".format(global_step))
                    saver.save(sess, filename)

                    metrics = evaluate_batch(config, model, config.val_num_batches,
                                             train_eval_file, sess, train_iterator)
                    write_metrics(metrics, writer, global_step, "train")

                    metrics = evaluate_batch(config, model, dev_total // config.batch_size + 1,
                                             dev_eval_file, sess, dev_iterator)
                    write_metrics(metrics, writer, global_step, "dev")

                    acc = metrics["accuracy"]
                    if acc > best_acc:
                        best_acc, best_ckpt = acc, global_step
                        save(config.best_ckpt, {"best_acc": str(acc), "best_ckpt": str(best_ckpt)},
                             config.best_ckpt)


def test(config):
    with open(config.dev_eval_file, "r") as fh:
        dev_eval_file = json.load(fh)
    with open(config.dev_meta, "r") as fh:
        meta = json.load(fh)

    dev_total = meta["total"]
    print("Building model...")
    parser = get_record_parser_qqp(config)
    graph = tf.Graph()
    with graph.as_default() as g:
        dev_dataset = get_dataset(config.dev_record_file, parser, config.test_batch_size)
        dev_iterator = dev_dataset.make_one_shot_iterator()
        model = QPCModel(config, graph=g)
        sess_config = tf.ConfigProto(allow_soft_placement=True)
        sess_config.gpu_options.allow_growth = True

        if os.path.exists(config.best_ckpt):
            with open(config.best_ckpt, "r") as fh:
                best_qqp_ckpt = json.load(fh)
                best_acc, best_ckpt = float(best_qqp_ckpt["best_acc"]), int(best_qqp_ckpt["best_ckpt"])

        with tf.Session(config=sess_config) as sess:
            writer = tf.summary.FileWriter(config.output_dir)
            checkpoint = "{}/model_{}.ckpt".format(config.output_dir, best_ckpt)
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            saver.restore(sess, checkpoint)
            global_step = sess.run(model.global_step)
            metrics = evaluate_batch(config, model, dev_total // config.test_batch_size + 1,
                                     dev_eval_file, sess, dev_iterator)
            print(metrics)
            write_metrics(metrics, writer, global_step, "test")


def evaluate_batch(config, model, num_batches, eval_file, sess, iterator):
    answer_dict = {}
    losses = []
    next_element = iterator.get_next()
    for _ in tqdm(range(1, num_batches + 1)):
        que1, que2, label, qa_id = sess.run(next_element)
        loss, pred_label = sess.run([model.loss, model.pred_label],
                                    feed_dict={
                                        model.que1: que1, model.que2: que2, model.label: label,
                                        model.dropout: config.dropout, model.qa_id: qa_id,
                                    })
        answer_dict_ = {}
        for qid, pl in zip(list(qa_id), list(np.argmax(pred_label, axis=1))):
            answer_dict_[str(qid)] = pl
        answer_dict.update(answer_dict_)
        losses.append(loss)
    loss = np.mean(losses)
    metrics = evaluate(eval_file, answer_dict)
    metrics["loss"] = loss
    return metrics


def evaluate(eval_file, answer_dict):
    correct = 0
    tp, p, t = 0, 0, 0
    for key, value in answer_dict.items():
        label = eval_file[key]["label"]
        pred_label = value
        p += pred_label
        t += label
        tp += pred_label * label
        correct += int(pred_label == label)
    if tp == 0:
        f1 = 0.
    else:
        prec = tp / p
        recall = tp / t
        f1 = (2 * prec * recall) / (prec + recall)
    return {'accuracy': correct / len(answer_dict), 'f1': f1}