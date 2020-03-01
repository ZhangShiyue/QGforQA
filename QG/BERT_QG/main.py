import os
import pickle as pkl
import numpy as np
import tensorflow as tf
import ujson as json
from tqdm import tqdm
import sys
sys.path.append('../..')

from LIB.bert import modeling
from LIB.utils import save, write_metrics, get_dataset, get_batch_dataset
from utils import get_record_parser
from model import BertEmb, QGModel, QGRLModel
from eval import convert_tokens_seq, evaluate_simple, evaluate


def train(config):
    with open(config.word_emb_file, "rb") as fh:
        word_mat = np.array(pkl.load(fh), dtype=np.float32)
    with open(config.pos_emb_file, "r") as fh:
        pos_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.ner_emb_file, "r") as fh:
        ner_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.label_emb_file, "r") as fh:
        label_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.word_dictionary, "r") as fh:
        word_dictionary = json.load(fh)
    with open(config.train_eval_file, "r") as fh:
        train_eval_file = json.load(fh)
    with open(config.dev_eval_file, "r") as fh:
        dev_eval_file = json.load(fh)
    with open(config.dev_meta, "r") as fh:
        meta = json.load(fh)
    with open(config.map_to_orig, "rb") as fh:
        map_to_orig = pkl.load(fh)

    id2word = {word_dictionary[w]: w for w in word_dictionary}
    dev_total = meta["total"]
    best_bleu, best_ckpt = 0., 0
    print("Building model...")
    parser = get_record_parser(config)
    graph_para = tf.Graph()
    graph_qg = tf.Graph()
    with graph_para.as_default() as g:
        train_dataset = get_batch_dataset(config.train_record_file, parser, config.batch_size)
        train_iterator = train_dataset.make_one_shot_iterator()
        dev_dataset = get_dataset(config.dev_record_file, parser, config.batch_size)
        dev_iterator = dev_dataset.make_one_shot_iterator()
        model_para = BertEmb(config, config.para_limit + 2, graph=g)
    with graph_qg.as_default() as g:
        model_qg = QGModel(config, word_mat, label_mat, pos_mat, ner_mat)
        model_qg.build_graph()
        model_qg.add_train_op()

    sess_para = tf.Session(graph=graph_para)
    sess_qg = tf.Session(graph=graph_qg)

    with sess_para.as_default():
        with graph_para.as_default():
            print("init from pretrained bert..")
            tvars = tf.trainable_variables()
            (assignment_map, initialized_variable_names) = \
                modeling.get_assignment_map_from_checkpoint(tvars, config.init_checkpoint)
            tf.train.init_from_checkpoint(config.init_checkpoint, assignment_map)
            sess_para.run(tf.global_variables_initializer())
    with sess_qg.as_default():
        with graph_qg.as_default():
            sess_qg.run(tf.global_variables_initializer())
            saver_qg = tf.train.Saver(max_to_keep=1000,
                                      var_list=[p for p in tf.global_variables() if "word_mat" not in p.name])
            if os.path.exists(os.path.join(config.output_dir, "checkpoint")):
                print(tf.train.latest_checkpoint(config.output_dir))
                saver_qg.restore(sess_qg, tf.train.latest_checkpoint(config.output_dir))
            if os.path.exists(config.best_ckpt):
                with open(config.best_ckpt, "r") as fh:
                    best_qg_ckpt = json.load(fh)
                    best_bleu, best_ckpt = float(best_qg_ckpt["best_bleu"]), int(best_qg_ckpt["best_ckpt"])
                    print("best_bleu:{}, best_ckpt:{}".format(best_bleu, best_ckpt))

    writer = tf.summary.FileWriter(config.output_dir)
    global_step = max(sess_qg.run(model_qg.global_step), 1)
    train_next_element = train_iterator.get_next()
    for _ in tqdm(range(global_step, config.num_steps + 1)):
        global_step = sess_qg.run(model_qg.global_step) + 1
        para, para_unk, ques, labels, pos_tags, ner_tags, qa_id = sess_para.run(train_next_element)
        para_emb = sess_para.run(model_para.bert_emb, feed_dict={model_para.input_ids: para_unk})
        loss, _ = sess_qg.run([model_qg.loss, model_qg.train_op], feed_dict={
            model_qg.para: para, model_qg.bert_para: para_emb, model_qg.que: ques,
            model_qg.labels: labels, model_qg.pos_tags: pos_tags, model_qg.ner_tags: ner_tags,
            model_qg.dropout: config.dropout, model_qg.qa_id: qa_id
        })
        if global_step % config.period == 0:
            loss_sum = tf.Summary(value=[tf.Summary.Value(
                    tag="model/loss", simple_value=loss), ])
            writer.add_summary(loss_sum, global_step)
        if global_step % config.checkpoint == 0:
            filename = os.path.join(config.output_dir, "model_{}.ckpt".format(global_step))
            saver_qg.save(sess_qg, filename)

            metrics = evaluate_batch(config, model_para, model_qg, sess_para, sess_qg, config.val_num_batches,
                                     train_eval_file, train_iterator, id2word, map_to_orig,
                                     evaluate_func=evaluate_simple)
            write_metrics(metrics, writer, global_step, "train")

            metrics = evaluate_batch(config, model_para, model_qg, sess_para, sess_qg,
                                     dev_total // config.batch_size + 1, dev_eval_file,
                                     dev_iterator, id2word, map_to_orig, evaluate_func=evaluate_simple)
            write_metrics(metrics, writer, global_step, "dev")
            bleu = metrics["bleu"]
            if bleu > best_bleu:
                best_bleu, best_ckpt = bleu, global_step
                save(config.best_ckpt, {"best_bleu": str(best_bleu), "best_ckpt": str(best_ckpt)},
                     config.best_ckpt)


def evaluate_batch(config, model_para, model_qg, sess_para, sess_qg, num_batches, eval_file, iterator,
                   id2word, map_to_orig, evaluate_func=evaluate):
    answer_dict = {}
    losses = []
    next_element = iterator.get_next()
    for _ in tqdm(range(1, num_batches + 1)):
        para, para_unk, ques, labels, pos_tags, ner_tags, qa_id = sess_para.run(next_element)
        para_emb = sess_para.run(model_para.bert_emb, feed_dict={model_para.input_ids: para_unk})
        loss, symbols, probs = sess_qg.run([model_qg.loss, model_qg.symbols, model_qg.probs],
                                           feed_dict={
                                               model_qg.para: para, model_qg.bert_para: para_emb,
                                               model_qg.que: ques, model_qg.labels: labels,
                                               model_qg.pos_tags: pos_tags,
                                               model_qg.ner_tags: ner_tags, model_qg.qa_id: qa_id,
                                               model_qg.temperature: config.temperature,
                                               model_qg.diverse_rate: config.diverse_rate
                                           })
        answer_dict_ = convert_tokens_seq(eval_file, qa_id, symbols, probs, id2word, map_to_orig)
        for key in answer_dict_:
            if key not in answer_dict:
                answer_dict[key] = answer_dict_[key]
        losses.append(loss)
    loss = np.mean(losses)
    metrics = evaluate_func(eval_file, answer_dict)
    metrics["loss"] = loss
    return metrics


def test(config):
    with open(config.word_emb_file, "rb") as fh:
        word_mat = np.array(pkl.load(fh), dtype=np.float32)
    with open(config.pos_emb_file, "r") as fh:
        pos_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.ner_emb_file, "r") as fh:
        ner_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.label_emb_file, "r") as fh:
        label_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.word_dictionary, "r") as fh:
        word_dictionary = json.load(fh)
    with open(config.test_eval_file, "r") as fh:
        test_eval_file = json.load(fh)
    with open(config.test_meta, "r") as fh:
        meta = json.load(fh)
    with open(config.map_to_orig, "rb") as fh:
        map_to_orig = pkl.load(fh)

    id2word = {word_dictionary[w]: w for w in word_dictionary}
    test_total = meta["total"]
    best_bleu, best_ckpt = 0., 0

    print("Building model...")
    parser = get_record_parser(config)
    graph_para = tf.Graph()
    graph_qg = tf.Graph()
    with graph_para.as_default() as g:
        test_dataset = get_dataset(config.test_record_file, parser, config.batch_size)
        test_iterator = test_dataset.make_one_shot_iterator()
        model_para = BertEmb(config, config.para_limit + 2, graph=g)
    with graph_qg.as_default() as g:
        model_qg = QGModel(config, word_mat, label_mat, pos_mat, ner_mat, trainable=False)
        model_qg.build_graph()
        model_qg.add_train_op()

    sess_para = tf.Session(graph=graph_para)
    sess_qg = tf.Session(graph=graph_qg)

    with sess_para.as_default():
        with graph_para.as_default():
            print("init from pretrained bert..")
            tvars = tf.trainable_variables()
            (assignment_map, initialized_variable_names) = \
                modeling.get_assignment_map_from_checkpoint(tvars, config.init_checkpoint)
            tf.train.init_from_checkpoint(config.init_checkpoint, assignment_map)
            sess_para.run(tf.global_variables_initializer())
    with sess_qg.as_default():
        with graph_qg.as_default():
            sess_qg.run(tf.global_variables_initializer())
            saver_qg = tf.train.Saver(max_to_keep=1000,
                                      var_list=[p for p in tf.global_variables() if "word_mat" not in p.name])
            if os.path.exists(config.best_ckpt):
                with open(config.best_ckpt, "r") as fh:
                    best_qg_ckpt = json.load(fh)
                    best_bleu, best_ckpt = float(best_qg_ckpt["best_bleu"]), int(best_qg_ckpt["best_ckpt"])
                    print("best_bleu:{}, best_ckpt:{}".format(best_bleu, best_ckpt))
            else:
                print("No best checkpoint!")
                exit()
            checkpoint = "{}/model_{}.ckpt".format(config.output_dir, best_ckpt)
            print(checkpoint)
            saver_qg.restore(sess_qg, checkpoint)
    writer = tf.summary.FileWriter(config.output_dir)
    metrics = evaluate_batch(config, model_para, model_qg, sess_para, sess_qg,
                             test_total // config.batch_size + 1, test_eval_file,
                             test_iterator, id2word, map_to_orig, evaluate_func=evaluate)
    print(metrics)
    write_metrics(metrics, writer, best_ckpt, "test")
