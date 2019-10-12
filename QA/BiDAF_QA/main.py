import os
import numpy as np
import ujson as json
import tensorflow as tf
from tqdm import tqdm
import sys
sys.path.append('../..')

from QG.ELMo_QG.utils import get_record_parser as get_record_parser_qg
from LIB.utils import get_dataset, get_batch_dataset, write_metrics, save
from model import BidafQA
from eval import convert_tokens_qa_for_qg, evaluate_qa_for_qg


def train_for_qg(config):
    with open(config.word_emb_file, "r") as fh:
        word_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.char_emb_file, "r") as fh:
        char_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.train_eval_file, "r") as fh:
        train_eval_file = json.load(fh)
    with open(config.dev_eval_file, "r") as fh:
        dev_eval_file = json.load(fh)
    with open(config.dev_meta, "r") as fh:
        meta = json.load(fh)

    dev_total = meta["total"]
    best_em, best_ckpt = 0., 0
    print("Building model...")
    parser = get_record_parser_qg(config)
    graph = tf.Graph()
    with graph.as_default() as g:
        train_record_file = config.train_record_file
        train_dataset = get_batch_dataset(train_record_file, parser, config.batch_size)
        dev_dataset = get_dataset(config.dev_record_file, parser, config.batch_size)
        train_iterator = train_dataset.make_one_shot_iterator()
        dev_iterator = dev_dataset.make_one_shot_iterator()

        model = BidafQA(config, word_mat, char_mat)

        sess_config = tf.ConfigProto(allow_soft_placement=True)
        sess_config.gpu_options.allow_growth = True

        with tf.Session(config=sess_config) as sess:
            # build graph
            model.build_graph()
            # add training operation
            model.add_train_op()
            writer = tf.summary.FileWriter(config.output_dir)
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(max_to_keep=1000,
                                   var_list=[p for p in tf.global_variables() if "word_mat" not in p.name])
            if os.path.exists(os.path.join(config.output_dir, "checkpoint")):
                saver.restore(sess, tf.train.latest_checkpoint(config.output_dir))
            if os.path.exists(config.best_ckpt):
                with open(config.best_ckpt, "r") as fh:
                    best_qa_ckpt = json.load(fh)
                    best_em, best_ckpt = float(best_qa_ckpt["best_em"]), int(best_qa_ckpt["best_ckpt"])
            global_step = max(sess.run(model.global_step), 1)
            train_next_element = train_iterator.get_next()
            for _ in tqdm(range(global_step, config.num_steps + 1)):
                global_step = sess.run(model.global_step) + 1
                _, para, para_char, _, que, que_char, _, _, _, _, _, _, y1, y2, qa_id = sess.run(train_next_element)
                loss, _ = sess.run([model.loss, model.train_op], feed_dict={
                    model.para: para, model.que: que, model.para_char: para_char, model.que_char: que_char,
                    model.y1: y1, model.y2: y2, model.dropout: config.dropout, model.qa_id: qa_id,
                })
                if global_step % config.period == 0:
                    loss_sum = tf.Summary(value=[tf.Summary.Value(
                            tag="model/loss", simple_value=loss), ])
                    writer.add_summary(loss_sum, global_step)
                if global_step % config.checkpoint == 0:
                    filename = os.path.join(
                            config.output_dir, "model_{}.ckpt".format(global_step))
                    saver.save(sess, filename)

                    metrics = evaluate_batch_qa_for_qg(model, config.val_num_batches,
                                                       train_eval_file, sess, train_iterator)
                    write_metrics(metrics, writer, global_step, "train")

                    metrics = evaluate_batch_qa_for_qg(model, dev_total // config.batch_size + 1,
                                                       dev_eval_file, sess, dev_iterator)
                    write_metrics(metrics, writer, global_step, "dev")

                    em = metrics["em"]
                    if em > best_em:
                        best_em, best_ckpt = em, global_step
                        save(config.best_ckpt, {"best_em": str(em), "best_ckpt": str(best_ckpt)},
                             config.best_ckpt)


def test_qa_for_qg(config):
    with open(config.word_emb_file, "r") as fh:
        word_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.char_emb_file, "r") as fh:
        char_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.dev_eval_file, "r") as fh:
        eval_file = json.load(fh)
    with open(config.dev_meta, "r") as fh:
        meta = json.load(fh)

    total = meta["total"]
    graph = tf.Graph()
    parser = get_record_parser_qg(config, is_test=True)
    print("Loading model...")
    with graph.as_default() as g:
        test_dataset = get_dataset(config.dev_record_file, parser, config.test_batch_size)
        test_iterator = test_dataset.make_one_shot_iterator()
        model = BidafQA(config, word_mat, char_mat, trainable=False)
        sess_config = tf.ConfigProto(allow_soft_placement=True)
        sess_config.gpu_options.allow_growth = True

        checkpoint_to_test = None
        if os.path.exists(config.best_ckpt):
            with open(config.best_ckpt, "r") as fh:
                best_qa_ckpt = json.load(fh)
                checkpoint_to_test = int(best_qa_ckpt["best_ckpt"])
        else:
            print("No best model to load!")
            exit()

        with tf.Session(config=sess_config) as sess:
            # build graph
            model.build_graph()
            # add training operation
            model.add_train_op()
            filename = "{}/beam{}".format(config.output_dir, config.beam_size)
            writer = tf.summary.FileWriter(filename)
            checkpoint = "{}/model_{}.ckpt".format(config.output_dir, checkpoint_to_test)
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(var_list=[p for p in tf.global_variables() if "word_mat" not in p.name])
            saver.restore(sess, checkpoint)
            global_step = sess.run(model.global_step)
            metrics = evaluate_batch_qa_for_qg(model, total // config.test_batch_size + 1, eval_file, sess,
                                               test_iterator)
            print(metrics)
            write_metrics(metrics, writer, global_step, "test")


def evaluate_batch_qa_for_qg(model, num_batches, eval_file, sess, iterator):
    answer_dict = {}
    losses = []
    next_element = iterator.get_next()
    for _ in tqdm(range(1, num_batches + 1)):
        _, para, para_char, _, que, que_char, _, _, _, _, _, _, y1, y2, qa_id = sess.run(next_element)
        loss, byp1, byp2, bprobs = sess.run([model.loss, model.byp1, model.byp2, model.bprobs],
                                            feed_dict={
                                                model.para: para, model.que: que,
                                                model.para_char: para_char, model.que_char: que_char,
                                                model.y1: y1, model.y2: y2, model.qa_id: qa_id,
                                            })
        answer_dict_ = convert_tokens_qa_for_qg(eval_file, qa_id, byp1, byp2, bprobs)
        answer_dict.update(answer_dict_)
        losses.append(loss)
    loss = np.mean(losses)
    metrics = evaluate_qa_for_qg(eval_file, answer_dict)
    metrics["loss"] = loss
    return metrics