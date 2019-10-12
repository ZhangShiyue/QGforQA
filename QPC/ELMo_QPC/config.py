"""
configurations
"""
import os
import tensorflow as tf
from prepare import prepare
from main import train, test


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
# processed data
flags.DEFINE_string("train_file", "data/processed/train/qqp", "Question pairs for training")
flags.DEFINE_string("dev_file", "data/processed/dev/qqp", "Question pairs for evalutation")
# vocab files
flags.DEFINE_string("vocab_file", "data/vocab/vocab.txt", "Vocabulary file")
flags.DEFINE_string("embedding_file", "data/vocab/elmo_token_embeddings.hdf5", "ElMo word embedding file")
flags.DEFINE_string("word_emb_file", "data/vocab/word_emb.json", "Glove word embedding file")
flags.DEFINE_string("word_dictionary", "data/vocab/word_dictionary.json", "Word dictionary")
# data parameters
flags.DEFINE_boolean("lower_word", True, "Whether to lower word")
flags.DEFINE_integer("ques_limit", 50, "Limit length for question")
# experimental data
flags.DEFINE_string("train_record_file", "data/experimental/train.tf_record", "Out file for train data")
flags.DEFINE_string("train_eval_file", "data/experimental/train_eval.json", "Out file for train eval")
flags.DEFINE_string("train_meta", "data/experimental/train_meta.json", "Out file for train meta")
flags.DEFINE_string("dev_record_file", "data/experimental/dev.tf_record", "Out file for dev data")
flags.DEFINE_string("dev_eval_file", "data/experimental/dev_eval.json", "Out file for dev eval")
flags.DEFINE_string("dev_meta", "data/experimental/dev_meta.json", "Out file for dev meta")
flags.DEFINE_string("test_record_file", "data/experimental/test.tf_record", "Out file for test data")
flags.DEFINE_string("test_eval_file", "data/experimental/test_eval.json", "Out file for test eval")
flags.DEFINE_string("test_meta", "data/experimental/test_meta.json", "Out file for test meta")
# experimental parameters
flags.DEFINE_integer("batch_size", 32, "Batch size")
flags.DEFINE_integer("test_batch_size", 32, "Batch size")
flags.DEFINE_integer("val_num_batches", 32, "Number of batches to evaluate the model")
flags.DEFINE_integer("qqp_hidden", 512, "Hidden size")
flags.DEFINE_boolean("use_elmo", True, "Use elmo or not")
# training settings
flags.DEFINE_integer("num_steps", 50000, "Number of steps")
flags.DEFINE_integer("period", 100, "period to save batch loss")
flags.DEFINE_integer("checkpoint", 1000, "checkpoint to save and evaluate the model")
flags.DEFINE_float("dropout", 0.3, "Dropout prob across the layers")
flags.DEFINE_float("ml_learning_rate", 0.001, "Learning rate")
flags.DEFINE_float("grad_clip", 5.0, "Global Norm gradient clipping rate")
flags.DEFINE_float("decay", 0.999, "Exponential moving average decay")
# training directory
flags.DEFINE_string("output_dir", "train/model_elmo_qpc", "Directory for tf event")
flags.DEFINE_string("best_ckpt", "train/model_elmo_qpc/best_ckpt.json", "The best checkpoint")


def main(_):
    config = flags.FLAGS
    if config.mode == "prepare":
        prepare(config)
    elif config.mode == "train":
        train(config)
    elif config.mode == "test":
        test(config)
    else:
        print("Unknown mode")
        exit(0)


if __name__ == "__main__":
    tf.app.run()