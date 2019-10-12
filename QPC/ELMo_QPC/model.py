import sys
sys.path.append('../..')
import tensorflow as tf
from LIB.layers import _linear
from bilm import BidirectionalLanguageModel, weight_layers


class QPCModel(object):
    def __init__(self, config, trainable=True, dev=False, graph=None):
        self.config = config
        self.graph = graph if graph is not None else tf.Graph()
        with self.graph.as_default():
            self.N = config.batch_size if (trainable or dev) else config.test_batch_size
            self.QL = config.ques_limit if (trainable or dev) else config.test_ques_limit

            self.global_step = tf.get_variable('global_step', shape=[], dtype=tf.int32,
                                               initializer=tf.constant_initializer(0), trainable=False)
            self.qa_id = tf.placeholder(tf.int32, [self.N], "qa_id")
            self.dropout = tf.placeholder_with_default(0.0, (), name="dropout")
            self.que1 = tf.placeholder(tf.int32, [self.N, self.QL + 2], "question1")
            self.que2 = tf.placeholder(tf.int32, [self.N, self.QL + 2], "question2")
            self.label = tf.placeholder(tf.int32, [self.N, 2], "label")

            # elmo
            self.bilm = BidirectionalLanguageModel(config.elmo_options_file, config.elmo_weight_file,
                                                   use_character_inputs=False,
                                                   embedding_weight_file=config.embedding_file)

        model = BiLSTMModel(self.que1, self.que2, self.label, self.bilm,
                            self.dropout, self.N, self.QL, config.qqp_hidden, True)
        self.loss, self.pred_label = model.build_model()
        _, pos_prob = tf.split(self.pred_label, [1, 1], axis=1)
        self.pos_prob = tf.reshape(pos_prob, [-1])

        if trainable:
            self.lr = config.ml_learning_rate
            self.opt = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.8, beta2=0.999, epsilon=1e-7)
            grads = self.opt.compute_gradients(self.loss)
            gradients, variables = zip(*grads)
            capped_grads, _ = tf.clip_by_global_norm(gradients, config.grad_clip)
            self.train_op = self.opt.apply_gradients(
                    zip(capped_grads, variables), global_step=self.global_step)


class BiLSTMModel(object):
    def __init__(self, question1, question2, label, bilm, dropout, batch_size,
                 ques_limit, hidden, use_elmo):
        self.que1 = question1
        _, self.que11, _ = tf.split(question1, [1, ques_limit, 1], 1)
        self.que2 = question2
        _, self.que21, _ = tf.split(question2, [1, ques_limit, 1], 1)
        self.label = label
        self.bilm = bilm
        self.use_elmo = use_elmo

        self.dropout = dropout
        self.N = batch_size
        self.QL = ques_limit
        self.d = hidden

        self.que1_mask = tf.cast(self.que11, tf.bool)
        self.que2_mask = tf.cast(self.que21, tf.bool)
        self.que1_len = tf.reduce_sum(tf.cast(self.que1_mask, tf.int32), axis=-1)
        self.que2_len = tf.reduce_sum(tf.cast(self.que2_mask, tf.int32), axis=-1)

        self.encoder_cells = [tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(hidden),
                              output_keep_prob=1.0 - dropout) for _ in range(4)]
        self.input_encoder_cells = [tf.nn.rnn_cell.MultiRNNCell(self.encoder_cells[:2]),
                                    tf.nn.rnn_cell.MultiRNNCell(self.encoder_cells[2:])]

    def build_model(self):
        q1_emb, q2_emb = self.elmo_input_embedding("input")
        q1, q2 = self.input_encoder(q1_emb, q2_emb)
        q1, q2 = tf.reduce_max(q1, 1), tf.reduce_max(q2, 1)
        input = [q1, q2, tf.abs(q1 - q2), q1 * q2]
        pred = self.output(input)
        batch_loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.label, logits=pred)
        return tf.reduce_mean(batch_loss), tf.nn.softmax(pred, axis=1)

    def output(self, input):
        with tf.variable_scope("Output_Layer"):
            hidden = tf.nn.relu(_linear(input, output_size=self.d, bias=True, scope="hidden"))
            pred = _linear(hidden, output_size=2, bias=True, scope="output")
            return pred

    def elmo_input_embedding(self, tag):
        que1_embeddings_op = self.bilm(self.que1)
        que2_embeddings_op = self.bilm(self.que2)
        elmo_que1 = weight_layers(tag, que1_embeddings_op, l2_coef=0.)['weighted_op']
        with tf.variable_scope('', reuse=True):
            elmo_que2 = weight_layers(tag, que2_embeddings_op, l2_coef=0.)['weighted_op']
        return elmo_que1, elmo_que2

    def input_encoder(self, c_emb, q_emb):
        with tf.variable_scope("Input_Encoder_Layer"):
            (ch_fw, ch_bw), _ = tf.nn.bidirectional_dynamic_rnn(self.input_encoder_cells[0],
                                                                self.input_encoder_cells[1],
                                                                c_emb, sequence_length=self.que1_len,
                                                                dtype='float', scope='input_encoder')
            c = tf.concat([ch_fw, ch_bw], axis=-1)
            (qh_fw, qh_bw), _ = tf.nn.bidirectional_dynamic_rnn(self.input_encoder_cells[0],
                                                                self.input_encoder_cells[1],
                                                                q_emb, sequence_length=self.que2_len,
                                                                dtype='float', scope='input_encoder')
            q = tf.concat([qh_fw, qh_bw], axis=-1)
            return c, q
