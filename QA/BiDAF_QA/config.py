"""
configurations
"""
import os
import tensorflow as tf
from main import train_for_qg, test_qa_for_qg


if not os.path.exists("data/vocab"):
    os.system("mkdir data/vocab")
if not os.path.exists("data/experimental"):
    os.system("mkdir data/experimental")
if not os.path.exists("train"):
    os.system("mkdir train")
flags = tf.flags
flags.DEFINE_string("mode", "train", "Running mode")
# common resources
flags.DEFINE_string("elmo_weight_file", "../../LIB/elmo/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5",
                    "Pretrained ElMo weight file")
flags.DEFINE_string("elmo_options_file", "../../LIB/elmo/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json",
                    "Pretrained ElMo option file")
flags.DEFINE_string("glove_word_file", "../../LIB/glove/glove.840B.300d.txt", "Glove word embedding source file")
# processed data
flags.DEFINE_string("train_para_file", "data/processed/train/paras", "Paragraphs for training")
flags.DEFINE_string("train_answer_file", "data/processed/train/answers", "Answers for training")
flags.DEFINE_string("train_question_file", "data/processed/train/questions", "Questions for training")
flags.DEFINE_string("dev_para_file", "data/processed/dev/paras", "Paragraphs for evalutation")
flags.DEFINE_string("dev_answer_file", "data/processed/dev/answers", "Answers for evalutation")
flags.DEFINE_string("dev_question_file", "data/processed/dev/questions", "Questions for evalutation")
flags.DEFINE_string("test_para_file", "data/processed/test/paras", "Paragraphs for testing")
flags.DEFINE_string("test_sent_file", "data/processed/test/sents", "Paragraphs for testing")
flags.DEFINE_string("test_answer_file", "data/processed/test/answers", "Answers for testing")
flags.DEFINE_string("test_question_file", "data/processed/test/questions", "Questions for testing")
# vocab files
flags.DEFINE_string("vocab_file", "data/vocab/vocab.txt", "Vocabulary file")
flags.DEFINE_string("embedding_file", "data/vocab/elmo_token_embeddings.hdf5", "ElMo word embedding file")
flags.DEFINE_string("word_emb_file", "data/vocab/word_emb.json", "Glove word embedding file")
flags.DEFINE_string("char_emb_file", "data/vocab/char_emb.json", "Char embedding file")
flags.DEFINE_string("pos_emb_file", "data/vocab/pos_emb.json", "Pos embedding file")
flags.DEFINE_string("ner_emb_file", "data/vocab/ner_emb.json", "Ner embedding file")
flags.DEFINE_string("label_emb_file", "data/vocab/label_emb.json", "Label embedding file")
flags.DEFINE_string("word_dictionary", "data/vocab/word_dictionary.json", "Word dictionary")
flags.DEFINE_string("char_dictionary", "data/vocab/char_dictionary.json", "Char dictionary")
flags.DEFINE_string("pos_dictionary", "data/vocab/pos_dictionary.json", "Pos dictionary")
flags.DEFINE_string("ner_dictionary", "data/vocab/ner_dictionary.json", "Ner dictionary")
flags.DEFINE_string("label_dictionary", "data/vocab/label_dictionary.json", "Label dictionary")
# data parameters
flags.DEFINE_integer("glove_word_size", int(2.2e6), "Corpus size for Glove")
flags.DEFINE_integer("glove_dim", 300, "Embedding dimension for Glove")
flags.DEFINE_integer("elmo_word_dim", 512, "Embedding dimension for ELMo")
flags.DEFINE_boolean("lower_word", True, "Whether to lower word")
flags.DEFINE_integer("vocab_size_limit", -1, "Vocab size")
flags.DEFINE_integer("char_dim", 64, "Embedding dimension for char")
flags.DEFINE_integer("para_limit", 400, "Limit length for paragraph")
flags.DEFINE_integer("ques_limit", 50, "Limit length for question")
flags.DEFINE_integer("ans_limit", 50, "Limit length for question")
flags.DEFINE_integer("test_para_limit", 400, "Limit length for paragraph in test file")
flags.DEFINE_integer("test_ques_limit", 50, "Limit length for question in test file")
flags.DEFINE_integer("test_ans_limit", 50, "Limit length for question in test file")
flags.DEFINE_integer("char_limit", 16, "Limit length for character")
# experimental data
flags.DEFINE_string("train_record_file", "data/experimental/train.tf_record", "Out file for train data")
flags.DEFINE_string("train_eval_file", "data/experimental/train_eval.json", "Out file for train eval")
flags.DEFINE_string("train_meta", "data/experimental/train_meta.json", "Out file for train meta")
flags.DEFINE_string("dev_record_file", "data/experimental/dev.tfrecords", "Out file for dev data")
flags.DEFINE_string("dev_eval_file", "data/experimental/dev_eval.json", "Out file for dev eval")
flags.DEFINE_string("dev_meta", "data/experimental/dev_meta.json", "Out file for dev meta")
flags.DEFINE_string("test_record_file", "data/experimental/test.tf_record", "Out file for test data")
flags.DEFINE_string("test_eval_file", "data/experimental/test_eval.json", "Out file for test eval")
flags.DEFINE_string("test_meta", "data/experimental/test_meta.json", "Out file for test meta")
# experimental parameters
flags.DEFINE_integer("batch_size", 64, "Batch size")
flags.DEFINE_integer("test_batch_size", 64, "Batch size")
flags.DEFINE_integer("val_num_batches", 64, "Number of batches to evaluate the model")
flags.DEFINE_integer("qa_hidden", 90, "Hidden size")
flags.DEFINE_boolean("use_elmo", True, "Use elmo or not")
flags.DEFINE_boolean("word_trainable", False, "Train word embeddings along or not")
flags.DEFINE_integer("beam_size", 10, "Beam size")
# training settings
flags.DEFINE_integer("num_steps", 20000, "Number of steps")
flags.DEFINE_integer("period", 100, "period to save batch loss")
flags.DEFINE_integer("checkpoint", 1000, "checkpoint to save and evaluate the model")
flags.DEFINE_float("dropout", 0.2, "Dropout prob across the layers")
flags.DEFINE_float("ml_learning_rate", 0.001, "Learning rate")
flags.DEFINE_float("grad_clip", 5.0, "Global Norm gradient clipping rate")
# training directory
flags.DEFINE_string("output_dir", "train/model_bidaf_qa_for_qg", "Directory for tf event")
flags.DEFINE_string("best_ckpt", "train/model_bidaf_qa_for_qg/best_ckpt.json", "The best checkpoint")


def main(_):
    config = flags.FLAGS
    if config.mode == "train_for_qg":
        train_for_qg(config)
    elif config.mode == "test_qa_for_qg":
        test_qa_for_qg(config)
    else:
        print("Unknown mode")
        exit(0)


if __name__ == "__main__":
    tf.app.run()