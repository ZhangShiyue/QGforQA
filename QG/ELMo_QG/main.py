import os
import h5py
import numpy as np
import ujson as json
import tensorflow as tf
from tqdm import tqdm
import sys
sys.path.append('../..')

from model import QGModel, QGRLModel
from eval import evaluate, evaluate_simple, convert_tokens, evaluate_rl, format_generated_ques_for_qpc, \
    format_generated_ques_for_qa
from LIB.utils import get_batch_dataset, get_dataset, write_metrics, save
from utils import get_record_parser
from QPC.ELMo_QPC.model import QPCModel
from QA.BiDAF_QA.model import BidafQA


def train(config):
    with open(config.word_emb_file, "r") as fh:
        word_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.pos_emb_file, "r") as fh:
        pos_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.ner_emb_file, "r") as fh:
        ner_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.label_emb_file, "r") as fh:
        label_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.train_eval_file, "r") as fh:
        train_eval_file = json.load(fh)
    with open(config.dev_eval_file, "r") as fh:
        dev_eval_file = json.load(fh)
    with open(config.dev_meta, "r") as fh:
        meta = json.load(fh)
    with open(config.word_dictionary, "r") as fh:
        word_dictionary = json.load(fh)
    with h5py.File(config.embedding_file, 'r') as fin:
        embed_weights = fin["embedding"][...]
        elmo_word_mat = np.zeros((embed_weights.shape[0] + 1, embed_weights.shape[1]), dtype=np.float32)
    elmo_word_mat[1:, :] = embed_weights

    id2word = {word_dictionary[w]: w for w in word_dictionary}
    dev_total = meta["total"]
    print("Building model...")
    parser = get_record_parser(config)
    graph = tf.Graph()
    best_bleu, best_ckpt = 0., 0
    with graph.as_default() as g:
        train_dataset = get_batch_dataset(config.train_record_file, parser, config.batch_size)
        dev_dataset = get_dataset(config.dev_record_file, parser, config.batch_size)
        train_iterator = train_dataset.make_one_shot_iterator()
        dev_iterator = dev_dataset.make_one_shot_iterator()

        model = QGModel(config, word_mat, elmo_word_mat, label_mat, pos_mat, ner_mat)
        model.build_graph()
        model.add_train_op()

        sess_config = tf.ConfigProto(allow_soft_placement=True)
        sess_config.gpu_options.allow_growth = True

        with tf.Session(config=sess_config) as sess:
            writer = tf.summary.FileWriter(config.output_dir)
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(max_to_keep=1000,
                                   var_list=[p for p in tf.global_variables() if "word_mat" not in p.name])
            if os.path.exists(os.path.join(config.output_dir, "checkpoint")):
                saver.restore(sess, tf.train.latest_checkpoint(config.output_dir))
            if os.path.exists(config.best_ckpt):
                with open(config.best_ckpt, "r") as fh:
                    best_qg_ckpt = json.load(fh)
                    best_bleu, best_ckpt = float(best_qg_ckpt["best_bleu"]), int(best_qg_ckpt["best_ckpt"])

            global_step = max(sess.run(model.global_step), 1)
            train_next_element = train_iterator.get_next()
            for _ in tqdm(range(global_step, config.num_steps + 1)):
                global_step = sess.run(model.global_step) + 1
                para, para_unk, para_char, que, que_unk, que_char, labels, pos_tags, ner_tags, \
                    que_labels, que_pos_tags, que_ner_tags, y1, y2, qa_id = sess.run(train_next_element)
                loss, _ = sess.run([model.loss, model.train_op], feed_dict={
                    model.para: para, model.para_unk: para_unk, model.que: que,
                    model.labels: labels, model.pos_tags: pos_tags, model.ner_tags: ner_tags,
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

                    metrics = evaluate_batch(config, model, config.val_num_batches, train_eval_file, sess,
                                             train_iterator, id2word, evaluate_func=evaluate_simple)
                    write_metrics(metrics, writer, global_step, "train")

                    metrics = evaluate_batch(config, model, dev_total // config.batch_size + 1, dev_eval_file,
                                             sess, dev_iterator, id2word, evaluate_func=evaluate_simple)
                    write_metrics(metrics, writer, global_step, "dev")

                    bleu = metrics["bleu"]
                    if bleu > best_bleu:
                        best_bleu, best_ckpt = bleu, global_step
                        save(config.best_ckpt, {"best_bleu": str(best_bleu), "best_ckpt": str(best_ckpt)},
                             config.best_ckpt)


def evaluate_batch(config, model, num_batches, eval_file, sess, iterator, id2word, evaluate_func=evaluate):
    answer_dict = {}
    losses = []
    next_element = iterator.get_next()
    for _ in tqdm(range(1, num_batches + 1)):
        para, para_unk, para_char, que, que_unk, que_char, labels, pos_tags, ner_tags, \
            que_labels, que_pos_tags, que_ner_tags, y1, y2, qa_id = sess.run(next_element)
        loss, symbols, probs = sess.run([model.loss, model.symbols, model.probs],
                                        feed_dict={
                                            model.para: para, model.para_unk: para_unk, model.que: que,
                                            model.labels: labels, model.pos_tags: pos_tags,
                                            model.ner_tags: ner_tags, model.qa_id: qa_id,
                                            model.temperature: config.temperature,
                                            model.diverse_rate: config.diverse_rate
                                        })
        answer_dict_ = convert_tokens(eval_file, qa_id, symbols, probs, id2word)
        for key in answer_dict_:
            if key not in answer_dict:
                answer_dict[key] = answer_dict_[key]
        losses.append(loss)
    loss = np.mean(losses)
    print(len(answer_dict))
    metrics = evaluate_func(eval_file, answer_dict)
    metrics["loss"] = loss
    return metrics


def test(config):
    with open(config.word_emb_file, "r") as fh:
        word_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.pos_emb_file, "r") as fh:
        pos_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.ner_emb_file, "r") as fh:
        ner_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.label_emb_file, "r") as fh:
        label_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.test_eval_file, "r") as fh:
        eval_file = json.load(fh)
    with open(config.test_meta, "r") as fh:
        meta = json.load(fh)
    with open(config.word_dictionary, "r") as fh:
        word_dictionary = json.load(fh)
    with h5py.File(config.embedding_file, 'r') as fin:
        embed_weights = fin["embedding"][...]
        elmo_word_mat = np.zeros((embed_weights.shape[0] + 1, embed_weights.shape[1]), dtype=np.float32)
    elmo_word_mat[1:, :] = embed_weights

    id2word = {word_dictionary[w]: w for w in word_dictionary}
    total = meta["total"]
    print(total)

    graph = tf.Graph()
    print("Loading model...")
    with graph.as_default() as g:
        test_iterator = get_dataset(config.test_record_file, get_record_parser(
                config, is_test=True), config.test_batch_size).make_one_shot_iterator()
        model = QGModel(config, word_mat, elmo_word_mat, label_mat, pos_mat, ner_mat, trainable=False)
        model.build_graph()
        model.add_train_op()
        sess_config = tf.ConfigProto(allow_soft_placement=True)
        sess_config.gpu_options.allow_growth = True

        if os.path.exists(config.best_ckpt):
            with open(config.best_ckpt, "r") as fh:
                best_ckpt = json.load(fh)
                checkpoint_to_test = int(best_ckpt["best_ckpt"])
        else:
            print("No Best!")
            exit()

        with tf.Session(config=sess_config) as sess:
            if config.diverse_beam:
                filename = "{}/diverse{}_beam{}".format(config.output_dir, config.diverse_rate, config.beam_size)
            elif config.sample:
                filename = "{}/temperature{}_sample{}".format(config.output_dir, config.temperature, config.sample_size)
            else:
                filename = "{}/beam{}".format(config.output_dir, config.beam_size)
            writer = tf.summary.FileWriter(filename)
            checkpoint = "{}/model_{}.ckpt".format(config.output_dir, checkpoint_to_test)
            print(checkpoint)
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(var_list=[p for p in tf.global_variables() if "word_mat" not in p.name])
            saver.restore(sess, checkpoint)
            global_step = sess.run(model.global_step)
            metrics = evaluate_batch(config, model, total // config.test_batch_size + 1, eval_file, sess,
                                     test_iterator, id2word)
            print(metrics)
            write_metrics(metrics, writer, global_step, "test")


def train_rl(config):
    with open(config.word_emb_file, "r") as fh:
        word_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.pos_emb_file, "r") as fh:
        pos_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.ner_emb_file, "r") as fh:
        ner_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.label_emb_file, "r") as fh:
        label_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.train_eval_file, "r") as fh:
        train_eval_file = json.load(fh)
    with open(config.dev_eval_file, "r") as fh:
        dev_eval_file = json.load(fh)
    with open(config.dev_meta, "r") as fh:
        meta = json.load(fh)
    with open(config.word_dictionary, "r") as fh:
        word_dictionary = json.load(fh)
    with h5py.File(config.embedding_file, 'r') as fin:
        embed_weights = fin["embedding"][...]
        elmo_word_mat = np.zeros((embed_weights.shape[0] + 1, embed_weights.shape[1]), dtype=np.float32)
    elmo_word_mat[1:, :] = embed_weights

    id2word = {word_dictionary[w]: w for w in word_dictionary}
    best_bleu, best_ckpt = 0., 0
    dev_total = meta["total"]
    print("Building model...")
    parser = get_record_parser(config)
    graph_qg = tf.Graph()
    with graph_qg.as_default():
        train_dataset = get_batch_dataset(config.train_record_file, parser, config.batch_size)
        dev_dataset = get_dataset(config.dev_record_file, parser, config.batch_size)
        train_iterator = train_dataset.make_one_shot_iterator()
        dev_iterator = dev_dataset.make_one_shot_iterator()

        model_qg = QGRLModel(config, word_mat, elmo_word_mat, label_mat, pos_mat, ner_mat)
        model_qg.build_graph()
        model_qg.add_train_op()

    sess_qg = tf.Session(graph=graph_qg)

    writer = tf.summary.FileWriter(config.output_dir)
    with sess_qg.as_default():
        with graph_qg.as_default():
            sess_qg.run(tf.global_variables_initializer())
            saver_qg = tf.train.Saver(max_to_keep=1000,
                                      var_list=[p for p in tf.global_variables() if "word_mat" not in p.name])
            if os.path.exists(os.path.join(config.output_dir, "checkpoint")):
                saver_qg.restore(sess_qg, tf.train.latest_checkpoint(config.output_dir))
            if os.path.exists(config.best_ckpt):
                with open(config.best_ckpt, "r") as fh:
                    best_qg_ckpt = json.load(fh)
                    best_bleu, best_ckpt = float(best_qg_ckpt["best_bleu"]), int(best_qg_ckpt["best_ckpt"])

    global_step = max(sess_qg.run(model_qg.global_step), 1)
    train_next_element = train_iterator.get_next()
    for _ in tqdm(range(global_step, config.num_steps + 1)):
        global_step = sess_qg.run(model_qg.global_step) + 1
        para, para_unk, para_char, que, que_unk, que_char, labels, pos_tags, ner_tags, \
            que_labels, que_pos_tags, que_ner_tags, y1, y2, qa_id = sess_qg.run(train_next_element)
        # get greedy search questions as baseline and sampled questions
        symbols, symbols_rl = sess_qg.run([model_qg.symbols, model_qg.symbols_rl], feed_dict={
            model_qg.para: para, model_qg.para_unk: para_unk, model_qg.que: que,
            model_qg.labels: labels, model_qg.pos_tags: pos_tags,
            model_qg.ner_tags: ner_tags, model_qg.qa_id: qa_id
        })
        # get rewards and format sampled questions
        reward, reward_rl, reward_base, que_rl = evaluate_rl(train_eval_file, qa_id, symbols, symbols_rl, id2word,
                                                             metric=config.rl_metric)
        # update model with policy gradient
        loss_ml, _ = sess_qg.run([model_qg.loss_ml, model_qg.train_op], feed_dict={
            model_qg.para: para, model_qg.para_unk: para_unk, model_qg.que: que,
            model_qg.labels: labels, model_qg.pos_tags: pos_tags,
            model_qg.ner_tags: ner_tags, model_qg.dropout: config.dropout, model_qg.qa_id: qa_id,
            model_qg.sampled_que: que_rl, model_qg.reward: reward
        })
        if global_step % config.period == 0:
            loss_sum = tf.Summary(value=[tf.Summary.Value(
                    tag="model/loss", simple_value=loss_ml), ])
            writer.add_summary(loss_sum, global_step)
            reward_base_sum = tf.Summary(value=[tf.Summary.Value(
                    tag="model/reward_base", simple_value=np.mean(reward_base)), ])
            writer.add_summary(reward_base_sum, global_step)
            reward_rl_sum = tf.Summary(value=[tf.Summary.Value(
                    tag="model/reward_rl", simple_value=np.mean(reward_rl)), ])
            writer.add_summary(reward_rl_sum, global_step)
        if global_step % config.checkpoint == 0:
            filename = os.path.join(
                    config.output_dir, "model_{}.ckpt".format(global_step))
            saver_qg.save(sess_qg, filename)

            metrics = evaluate_batch(config, model_qg, config.val_num_batches, train_eval_file, sess_qg,
                                     train_iterator, id2word, evaluate_func=evaluate_simple)
            write_metrics(metrics, writer, global_step, "train")

            metrics = evaluate_batch(config, model_qg, dev_total // config.batch_size + 1, dev_eval_file,
                                     sess_qg, dev_iterator, id2word, evaluate_func=evaluate_simple)
            write_metrics(metrics, writer, global_step, "dev")

            bleu = metrics["bleu"]
            if bleu > best_bleu:
                best_bleu, best_ckpt = bleu, global_step
                save(config.best_ckpt, {"best_bleu": str(best_bleu), "best_ckpt": str(best_ckpt)},
                     config.best_ckpt)


def train_qpp(config):
    with open(config.word_emb_file, "r") as fh:
        word_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.pos_emb_file, "r") as fh:
        pos_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.ner_emb_file, "r") as fh:
        ner_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.label_emb_file, "r") as fh:
        label_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.train_eval_file, "r") as fh:
        train_eval_file = json.load(fh)
    with open(config.dev_eval_file, "r") as fh:
        dev_eval_file = json.load(fh)
    with open(config.dev_meta, "r") as fh:
        meta = json.load(fh)
    with open(config.word_dictionary, "r") as fh:
        word_dictionary = json.load(fh)
    with h5py.File(config.embedding_file, 'r') as fin:
        embed_weights = fin["embedding"][...]
        elmo_word_mat = np.zeros((embed_weights.shape[0] + 1, embed_weights.shape[1]), dtype=np.float32)
    elmo_word_mat[1:, :] = embed_weights

    id2word = {word_dictionary[w]: w for w in word_dictionary}
    dev_total = meta["total"]
    best_bleu, best_ckpt = 0., 0
    print("Building model...")
    parser = get_record_parser(config)
    graph_qg = tf.Graph()
    graph_qqp = tf.Graph()
    with graph_qg.as_default():
        train_dataset = get_batch_dataset(config.train_record_file, parser, config.batch_size)
        dev_dataset = get_dataset(config.dev_record_file, parser, config.batch_size)
        train_iterator = train_dataset.make_one_shot_iterator()
        dev_iterator = dev_dataset.make_one_shot_iterator()

    with graph_qg.as_default() as g:
        model_qg = QGRLModel(config, word_mat, elmo_word_mat, label_mat, pos_mat, ner_mat)
        model_qg.build_graph()
        model_qg.add_train_op()
    with graph_qqp.as_default() as g:
        model_qqp = QPCModel(config, dev=True, trainable=False, graph=g)

    sess_qg = tf.Session(graph=graph_qg)
    sess_qqp = tf.Session(graph=graph_qqp)

    writer = tf.summary.FileWriter(config.output_dir)
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
    with sess_qqp.as_default():
        with graph_qqp.as_default():
            sess_qqp.run(tf.global_variables_initializer())
            saver_qqp = tf.train.Saver()
            if os.path.exists(config.best_ckpt_qpc):
                with open(config.best_ckpt_qpc, "r") as fh:
                    best_qpc_ckpt = json.load(fh)
                    best_ckpt = int(best_qpc_ckpt["best_ckpt"])
                print("{}/model_{}.ckpt".format(config.output_dir_qpc, best_ckpt))
                saver_qqp.restore(sess_qqp, "{}/model_{}.ckpt".format(config.output_dir_qpc, best_ckpt))
            else:
                print("NO the best QPC model to load!")
                exit()

    global_step = max(sess_qg.run(model_qg.global_step), 1)
    train_next_element = train_iterator.get_next()
    for _ in tqdm(range(global_step, config.num_steps + 1)):
        global_step = sess_qg.run(model_qg.global_step) + 1
        para, para_unk, para_char, que, que_unk, que_char, labels, pos_tags, ner_tags, \
            que_labels, que_pos_tags, que_ner_tags, y1, y2, qa_id = sess_qg.run(train_next_element)
        symbols, symbols_rl = sess_qg.run([model_qg.symbols, model_qg.symbols_rl], feed_dict={
            model_qg.para: para, model_qg.para_unk: para_unk, model_qg.que: que, model_qg.labels: labels,
            model_qg.pos_tags: pos_tags, model_qg.ner_tags: ner_tags, model_qg.qa_id: qa_id
        })
        # format questions for QPC
        que_base, que_unk_base, que_rl, que_unk_rl = \
            format_generated_ques_for_qpc(qa_id, symbols, symbols_rl, config.batch_size,
                                          config.ques_limit, id2word)
        label = np.zeros((config.batch_size, 2), dtype=np.int32)
        # QQP reward
        reward_base = sess_qqp.run(model_qqp.pos_prob, feed_dict={
            model_qqp.que1: que_unk, model_qqp.que2: que_unk_base,
            model_qqp.label: label, model_qqp.qa_id: qa_id,
        })
        reward_rl = sess_qqp.run(model_qqp.pos_prob, feed_dict={
            model_qqp.que1: que_unk, model_qqp.que2: que_unk_rl,
            model_qqp.label: label, model_qqp.qa_id: qa_id,
        })
        reward = [rr - rb for rr, rb in zip(reward_rl, reward_base)]
        # train with rl
        loss_ml, _ = sess_qg.run([model_qg.loss_ml, model_qg.train_op], feed_dict={
            model_qg.para: para, model_qg.para_unk: para_unk, model_qg.labels: labels,
            model_qg.pos_tags: pos_tags, model_qg.ner_tags: ner_tags,
            model_qg.dropout: config.dropout, model_qg.qa_id: qa_id,
            model_qg.que: que, model_qg.sampled_que: que_rl, model_qg.reward: reward
        })
        if global_step % config.period == 0:
            loss_sum = tf.Summary(value=[tf.Summary.Value(
                    tag="model/loss", simple_value=loss_ml), ])
            writer.add_summary(loss_sum, global_step)
            reward_base_sum = tf.Summary(value=[tf.Summary.Value(
                    tag="model/reward_base", simple_value=np.mean(reward_base)), ])
            writer.add_summary(reward_base_sum, global_step)
            reward_rl_sum = tf.Summary(value=[tf.Summary.Value(
                    tag="model/reward_rl", simple_value=np.mean(reward_rl)), ])
            writer.add_summary(reward_rl_sum, global_step)
        if global_step % config.checkpoint == 0:
            filename = os.path.join(
                    config.output_dir, "model_{}.ckpt".format(global_step))
            saver_qg.save(sess_qg, filename)

            metrics = evaluate_batch(config, model_qg, config.val_num_batches, train_eval_file, sess_qg,
                                     train_iterator, id2word, evaluate_func=evaluate_simple)
            write_metrics(metrics, writer, global_step, "train")

            metrics = evaluate_batch(config, model_qg, dev_total // config.batch_size + 1, dev_eval_file,
                                     sess_qg, dev_iterator, id2word, evaluate_func=evaluate_simple)
            write_metrics(metrics, writer, global_step, "dev")

            bleu = metrics["bleu"]
            if bleu > best_bleu:
                best_bleu, best_ckpt = bleu, global_step
                save(config.best_ckpt, {"best_bleu": str(best_bleu), "best_ckpt": str(best_ckpt)},
                     config.best_ckpt)


def train_qap(config):
    with open(config.word_emb_file, "r") as fh:
        word_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.char_emb_file, "r") as fh:
        char_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.pos_emb_file, "r") as fh:
        pos_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.ner_emb_file, "r") as fh:
        ner_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.label_emb_file, "r") as fh:
        label_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.train_eval_file, "r") as fh:
        train_eval_file = json.load(fh)
    with open(config.dev_eval_file, "r") as fh:
        dev_eval_file = json.load(fh)
    with open(config.dev_meta, "r") as fh:
        meta = json.load(fh)
    with open(config.word_dictionary, "r") as fh:
        word_dictionary = json.load(fh)
    with open(config.char_dictionary, "r") as fh:
        char_dictionary = json.load(fh)
    with h5py.File(config.embedding_file, 'r') as fin:
        embed_weights = fin["embedding"][...]
        elmo_word_mat = np.zeros((embed_weights.shape[0] + 1, embed_weights.shape[1]), dtype=np.float32)
    elmo_word_mat[1:, :] = embed_weights

    id2word = {word_dictionary[w]: w for w in word_dictionary}
    dev_total = meta["total"]
    best_bleu, best_ckpt = 0., 0
    print("Building model...")
    parser = get_record_parser(config)
    graph_qg = tf.Graph()
    graph_qa = tf.Graph()
    with graph_qg.as_default():
        train_dataset = get_batch_dataset(config.train_record_file, parser, config)
        dev_dataset = get_dataset(config.dev_record_file, parser, config)
        train_iterator = train_dataset.make_one_shot_iterator()
        dev_iterator = dev_dataset.make_one_shot_iterator()

    with graph_qg.as_default() as g:
        model_qg = QGRLModel(config, word_mat, elmo_word_mat, label_mat, pos_mat, ner_mat)
    with graph_qa.as_default() as g:
        model_qa = BidafQA(config, word_mat, char_mat, dev=True, trainable=False)

    sess_qg = tf.Session(graph=graph_qg)
    sess_qa = tf.Session(graph=graph_qa)

    writer = tf.summary.FileWriter(config.output_dir)
    with sess_qg.as_default():
        with graph_qg.as_default():
            model_qg.build_graph()
            model_qg.add_train_op()
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
    with sess_qa.as_default():
        with graph_qa.as_default():
            model_qa.build_graph()
            model_qa.add_train_op()
            sess_qa.run(tf.global_variables_initializer())
            saver_qa = tf.train.Saver(max_to_keep=1000,
                                      var_list=[p for p in tf.global_variables() if "word_mat" not in p.name])
            if os.path.exists(config.best_ckpt_qa):
                with open(config.best_ckpt_qa, "r") as fh:
                    best_qpc_ckpt = json.load(fh)
                    best_ckpt = int(best_qpc_ckpt["best_ckpt"])
                print("{}/model_{}.ckpt".format(config.output_dir_qa, best_ckpt))
                saver_qa.restore(sess_qa, "{}/model_{}.ckpt".format(config.output_dir_qa, best_ckpt))
            else:
                print("NO the best QA model to load!")
                exit()

    global_step = max(sess_qg.run(model_qg.global_step), 1)
    train_next_element = train_iterator.get_next()
    for _ in tqdm(range(global_step, config.num_steps + 1)):
        global_step = sess_qg.run(model_qg.global_step) + 1
        para, para_unk, para_char, que, que_unk, que_char, labels, pos_tags, ner_tags, \
            que_labels, que_pos_tags, que_ner_tags, y1, y2, qa_id = sess_qg.run(train_next_element)
        symbols, symbols_rl = sess_qg.run([model_qg.symbols, model_qg.symbols_rl], feed_dict={
            model_qg.para: para, model_qg.para_unk: para_unk, model_qg.que: que, model_qg.labels: labels,
            model_qg.pos_tags: pos_tags, model_qg.ner_tags: ner_tags, model_qg.qa_id: qa_id
        })
        # format questions for QA
        que_base, que_unk_base, que_char_base, que_rl, que_unk_rl, que_char_rl = \
            format_generated_ques_for_qa(train_eval_file, qa_id, symbols, symbols_rl, config.batch_size,
                                         config.ques_limit, config.char_limit, id2word, char_dictionary)
        # QAP reward
        base_qa_loss = sess_qa.run(model_qa.batch_loss,
                                   feed_dict={
                                       model_qa.para: para_unk, model_qa.para_char: para_char,
                                       model_qa.que: que_unk_base, model_qa.que_char: que_char_base,
                                       model_qa.y1: y1, model_qa.y2: y2, model_qa.qa_id: qa_id,
                                   })
        qa_loss = sess_qa.run(model_qa.batch_loss,
                              feed_dict={
                                model_qa.para: para_unk, model_qa.para_char: para_char,
                                model_qa.que: que_unk_rl, model_qa.que_char: que_char_rl,
                                model_qa.y1: y1, model_qa.y2: y2, model_qa.qa_id: qa_id,
                              })
        reward_base = list(map(lambda x: np.exp(-x), list(base_qa_loss)))
        reward_rl = list(map(lambda x: np.exp(-x), list(qa_loss)))
        reward = list(map(lambda x: x[0] - x[1], zip(reward_rl, reward_base)))
        # train with rl
        loss_ml, _ = sess_qg.run([model_qg.loss_ml, model_qg.train_op], feed_dict={
            model_qg.para: para, model_qg.para_unk: para_unk, model_qg.labels: labels,
            model_qg.pos_tags: pos_tags, model_qg.ner_tags: ner_tags,
            model_qg.dropout: config.dropout, model_qg.qa_id: qa_id,
            model_qg.que: que, model_qg.sampled_que: que_rl, model_qg.reward: reward
        })
        if global_step % config.period == 0:
            loss_sum = tf.Summary(value=[tf.Summary.Value(
                    tag="model/loss", simple_value=loss_ml), ])
            writer.add_summary(loss_sum, global_step)
            reward_base_sum = tf.Summary(value=[tf.Summary.Value(
                    tag="model/reward_base", simple_value=np.mean(reward_base)), ])
            writer.add_summary(reward_base_sum, global_step)
            reward_rl_sum = tf.Summary(value=[tf.Summary.Value(
                    tag="model/reward_rl", simple_value=np.mean(reward_rl)), ])
            writer.add_summary(reward_rl_sum, global_step)
        if global_step % config.checkpoint == 0:
            filename = os.path.join(
                    config.qg_save_dir, "model_{}.ckpt".format(global_step))
            saver_qg.save(sess_qg, filename)

            metrics = evaluate_batch(config, model_qg, config.val_num_batches,
                                     train_eval_file, sess_qg, train_iterator, id2word,
                                     evaluate_func=evaluate_simple)
            write_metrics(metrics, writer, global_step, "train")

            metrics = evaluate_batch(config, model_qg, dev_total // config.batch_size + 1,
                                     dev_eval_file, sess_qg, dev_iterator, id2word,
                                     evaluate_func=evaluate_simple)
            write_metrics(metrics, writer, global_step, "dev")

            bleu = metrics["bleu"]
            if bleu > best_bleu:
                best_bleu, best_ckpt = bleu, global_step
                save(config.best_qg_ckpt, {"best_bleu": str(best_bleu), "best_ckpt": str(best_ckpt)},
                     config.best_qg_ckpt)


def train_qqp_qap(config):
    with open(config.word_emb_file, "r") as fh:
        word_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.char_emb_file, "r") as fh:
        char_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.pos_emb_file, "r") as fh:
        pos_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.ner_emb_file, "r") as fh:
        ner_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.label_emb_file, "r") as fh:
        label_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.train_eval_file, "r") as fh:
        train_eval_file = json.load(fh)
    with open(config.dev_eval_file, "r") as fh:
        dev_eval_file = json.load(fh)
    with open(config.dev_meta, "r") as fh:
        meta = json.load(fh)
    with open(config.word_dictionary, "r") as fh:
        word_dictionary = json.load(fh)
    with open(config.char_dictionary, "r") as fh:
        char_dictionary = json.load(fh)
    with h5py.File(config.embedding_file, 'r') as fin:
        embed_weights = fin["embedding"][...]
        elmo_word_mat = np.zeros((embed_weights.shape[0] + 1, embed_weights.shape[1]), dtype=np.float32)
    elmo_word_mat[1:, :] = embed_weights

    id2word = {word_dictionary[w]: w for w in word_dictionary}
    dev_total = meta["total"]
    best_bleu, best_ckpt = 0., 0
    print("Building model...")
    parser = get_record_parser(config)
    graph_qg = tf.Graph()
    graph_qqp = tf.Graph()
    graph_qa = tf.Graph()
    with graph_qg.as_default():
        train_dataset = get_batch_dataset(config.train_record_file, parser, config)
        dev_dataset = get_dataset(config.dev_record_file, parser, config)
        train_iterator = train_dataset.make_one_shot_iterator()
        dev_iterator = dev_dataset.make_one_shot_iterator()

    with graph_qg.as_default() as g:
        model_qg = QGRLModel(config, word_mat, elmo_word_mat, label_mat, pos_mat, ner_mat)
        model_qg.build_graph()
        model_qg.add_train_op()
    with graph_qqp.as_default() as g:
        model_qqp = QPCModel(config, dev=True, trainable=False, graph=g)
    with graph_qa.as_default() as g:
        model_qa = BidafQA(config, word_mat, char_mat, dev=True, trainable=False)

    sess_qg = tf.Session(graph=graph_qg)
    sess_qqp = tf.Session(graph=graph_qqp)
    sess_qa = tf.Session(graph=graph_qa)

    writer = tf.summary.FileWriter(config.output_dir)
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
    with sess_qqp.as_default():
        with graph_qqp.as_default():
            sess_qqp.run(tf.global_variables_initializer())
            saver_qqp = tf.train.Saver(max_to_keep=1000,
                                       var_list=[p for p in tf.global_variables() if "word_mat" not in p.name])
            if os.path.exists(config.best_ckpt_qpc):
                with open(config.best_ckpt_qpc, "r") as fh:
                    best_qpc_ckpt = json.load(fh)
                    best_ckpt = int(best_qpc_ckpt["best_ckpt"])
                print("{}/model_{}.ckpt".format(config.output_dir_qpc, best_ckpt))
                saver_qqp.restore(sess_qqp, "{}/model_{}.ckpt".format(config.output_dir_qpc, best_ckpt))
            else:
                print("NO the best QPC model to load!")
                exit()
    with sess_qa.as_default():
        with graph_qa.as_default():
            model_qa.build_graph()
            model_qa.add_train_op()
            sess_qa.run(tf.global_variables_initializer())
            saver_qa = tf.train.Saver(max_to_keep=1000,
                                      var_list=[p for p in tf.global_variables() if "word_mat" not in p.name])
            if os.path.exists(config.best_ckpt_qa):
                with open(config.best_ckpt_qa, "r") as fh:
                    best_qpc_ckpt = json.load(fh)
                    best_ckpt = int(best_qpc_ckpt["best_ckpt"])
                print("{}/model_{}.ckpt".format(config.output_dir_qa, best_ckpt))
                saver_qa.restore(sess_qa, "{}/model_{}.ckpt".format(config.output_dir_qa, best_ckpt))
            else:
                print("NO the best QA model to load!")
                exit()

    global_step = max(sess_qg.run(model_qg.global_step), 1)
    train_next_element = train_iterator.get_next()
    for _ in tqdm(range(global_step, config.num_steps + 1)):
        global_step = sess_qg.run(model_qg.global_step) + 1
        para, para_unk, para_char, que, que_unk, que_char, labels, pos_tags, ner_tags, \
            que_labels, que_pos_tags, que_ner_tags, y1, y2, qa_id = sess_qg.run(train_next_element)
        symbols, symbols_rl = sess_qg.run([model_qg.symbols, model_qg.symbols_rl], feed_dict={
            model_qg.para: para, model_qg.para_unk: para_unk, model_qg.que: que, model_qg.labels: labels,
            model_qg.pos_tags: pos_tags, model_qg.ner_tags: ner_tags, model_qg.qa_id: qa_id
        })
        # format sample for QA
        que_base, que_unk_base, que_char_base, que_rl, que_unk_rl, que_char_rl = \
            format_generated_ques_for_qa(train_eval_file, qa_id, symbols, symbols_rl, config.batch_size,
                                         config.ques_limit, config.char_limit, id2word, char_dictionary)
        label = np.zeros((config.batch_size, 2), dtype=np.int32)
        if global_step % 4 == 0:
            # QPP reward
            reward_base = sess_qqp.run(model_qqp.pos_prob, feed_dict={
                model_qqp.que1: que_unk, model_qqp.que2: que_unk_base,
                model_qqp.label: label, model_qqp.qa_id: qa_id,
            })
            reward_rl = sess_qqp.run(model_qqp.pos_prob, feed_dict={
                model_qqp.que1: que_unk, model_qqp.que2: que_unk_rl,
                model_qqp.label: label, model_qqp.qa_id: qa_id,
            })
            reward = [rr - rb for rr, rb in zip(reward_rl, reward_base)]
            mixing_ratio = 0.99
        else:
            # QAP reward
            base_qa_loss = sess_qa.run(model_qa.batch_loss,
                                       feed_dict={
                                           model_qa.para: para_unk, model_qa.para_char: para_char,
                                           model_qa.que: que_unk_base, model_qa.que_char: que_char_base,
                                           model_qa.y1: y1, model_qa.y2: y2, model_qa.qa_id: qa_id,
                                       })
            qa_loss = sess_qa.run(model_qa.batch_loss,
                                  feed_dict={
                                      model_qa.para: para_unk, model_qa.para_char: para_char,
                                      model_qa.que: que_unk_rl, model_qa.que_char: que_char_rl,
                                      model_qa.y1: y1, model_qa.y2: y2, model_qa.qa_id: qa_id,
                                  })
            reward_base = list(map(lambda x: np.exp(-x), list(base_qa_loss)))
            reward_rl = list(map(lambda x: np.exp(-x), list(qa_loss)))
            reward = list(map(lambda x: x[0] - x[1], zip(reward_rl, reward_base)))
            mixing_ratio = 0.97
        # train with rl
        loss_ml, _ = sess_qg.run([model_qg.loss_ml, model_qg.train_op], feed_dict={
            model_qg.para: para, model_qg.para_unk: para_unk, model_qg.labels: labels,
            model_qg.pos_tags: pos_tags, model_qg.ner_tags: ner_tags, model_qg.que: que,
            model_qg.dropout: config.dropout, model_qg.qa_id: qa_id,
            model_qg.sampled_que: que_rl, model_qg.reward: reward, model_qg.lamda: mixing_ratio
        })
        if global_step % config.period == 0:
            loss_sum = tf.Summary(value=[tf.Summary.Value(
                    tag="model/loss", simple_value=loss_ml), ])
            writer.add_summary(loss_sum, global_step)
            reward_base_sum = tf.Summary(value=[tf.Summary.Value(
                    tag="model/reward_base", simple_value=np.mean(reward_base)), ])
            writer.add_summary(reward_base_sum, global_step)
            reward_rl_sum = tf.Summary(value=[tf.Summary.Value(
                    tag="model/reward_rl", simple_value=np.mean(reward_rl)), ])
            writer.add_summary(reward_rl_sum, global_step)
        if global_step % config.checkpoint == 0:
            filename = os.path.join(
                    config.output_dir, "model_{}.ckpt".format(global_step))
            saver_qg.save(sess_qg, filename)

            metrics = evaluate_batch(config, model_qg, config.val_num_batches, train_eval_file, sess_qg,
                                     train_iterator, id2word, evaluate_func=evaluate_simple)
            write_metrics(metrics, writer, global_step, "train")

            metrics = evaluate_batch(config, model_qg, dev_total // config.batch_size + 1, dev_eval_file,
                                     sess_qg, dev_iterator, id2word, evaluate_func=evaluate_simple)
            write_metrics(metrics, writer, global_step, "dev")

            bleu = metrics["bleu"]
            if bleu > best_bleu:
                best_bleu, best_ckpt = bleu, global_step
                save(config.best_ckpt, {"best_bleu": str(best_bleu), "best_ckpt": str(best_ckpt)},
                     config.best_ckpt)