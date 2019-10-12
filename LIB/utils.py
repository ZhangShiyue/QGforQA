import ujson as json
import tensorflow as tf


def save(filename, obj, message=None):
    if message is not None:
        print("Saving {}...".format(message))
        with open(filename, "w") as fh:
            json.dump(obj, fh)


def get_batch_dataset(record_file, parser, batch_size):
    print(record_file)
    num_threads = tf.constant(4, dtype=tf.int32)
    dataset = tf.data.TFRecordDataset(record_file).map(
            parser, num_parallel_calls=num_threads).shuffle(100).repeat()
    dataset = dataset.batch(batch_size)
    return dataset


def get_dataset(record_file, parser, batch_size):
    print(record_file)
    num_threads = tf.constant(4, dtype=tf.int32)
    dataset = tf.data.TFRecordDataset(record_file).map(
            parser, num_parallel_calls=num_threads).repeat().batch(batch_size)
    return dataset


def write_metrics(metrics, writer, global_step, tag):
    for key in metrics:
        key_sum = tf.Summary(value=[tf.Summary.Value(
                tag="{}/{}".format(tag, key), simple_value=metrics[key]), ])
        writer.add_summary(key_sum, global_step)
    writer.flush()
