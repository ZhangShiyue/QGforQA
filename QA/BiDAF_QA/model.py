import tensorflow as tf
from bilm import BidirectionalLanguageModel, weight_layers
import sys
sys.path.append('../..')

from LIB.tf_utils.ops import bi_cudnn_rnn_encoder, bidaf_attention, tri_linear_attention, \
    self_attention
from LIB.layers import mask_logits, conv


class BidafQA(object):
    def __init__(self, config, word_mat, char_mat, mix=False, dev=False, trainable=True):
        self.config = config
        self.trainable = trainable
        self.N = (config.batch_size * 2 if mix else config.batch_size) if (trainable or dev) else config.test_batch_size
        self.PL = config.para_limit if (trainable or dev) else config.test_para_limit
        self.QL = config.ques_limit if (trainable or dev) else config.test_ques_limit
        self.AL = config.ans_limit if (trainable or dev) else config.test_ans_limit
        self.CL = config.char_limit

        self.d = config.qa_hidden
        self.dc = config.char_dim

        self.global_step = tf.get_variable('global_step', shape=[], dtype=tf.int32,
                                           initializer=tf.constant_initializer(0), trainable=False)
        self.qa_id = tf.placeholder(tf.int32, [self.N], "qa_id")
        self.dropout = tf.placeholder_with_default(0.0, (), name="dropout")
        self.para = tf.placeholder(tf.int32, [self.N, self.PL + 2], "paragraph")
        self.para_char = tf.placeholder(tf.int32, [self.N, self.PL, self.CL], "paragraph_char")
        self.que = tf.placeholder(tf.int32, [self.N, self.QL + 2], "question")
        self.que_char = tf.placeholder(tf.int32, [self.N, self.QL, self.CL], "question_char")
        self.y1 = tf.placeholder(tf.int32, [self.N, self.PL], "answer_index1")
        self.y2 = tf.placeholder(tf.int32, [self.N, self.PL], "answer_index2")
        self.labels = tf.placeholder_with_default(tf.ones([self.N], dtype=tf.int32), (self.N), name="labels")

        _, self.para1, _ = tf.split(self.para, [1, self.PL, 1], axis=1)
        _, self.que1, _ = tf.split(self.que, [1, self.QL, 1], axis=1)
        self.para_mask = tf.cast(self.para1, tf.bool)
        self.que_mask = tf.cast(self.que1, tf.bool)
        self.para_len = tf.reduce_sum(tf.cast(self.para_mask, tf.int32), axis=-1)
        self.que_len = tf.reduce_sum(tf.cast(self.que_mask, tf.int32), axis=-1)

        with tf.device("/cpu:0"):
            self.word_mat = tf.get_variable("word_mat", initializer=tf.constant(word_mat, dtype=tf.float32),
                                            trainable=config.word_trainable)
            self.char_mat = tf.get_variable("char_mat", initializer=tf.constant(char_mat, dtype=tf.float32),
                                            trainable=True)
        # elmo
        self.elmo_bilm = BidirectionalLanguageModel(config.elmo_options_file, config.elmo_weight_file,
                                                    use_character_inputs=False,
                                                    embedding_weight_file=config.embedding_file)

    def build_graph(self):
        para_embeddings_op = self.elmo_bilm(self.para)
        que_embeddings_op = self.elmo_bilm(self.que)
        with tf.variable_scope('elmo_encodings_input'):
            elmo_para_input = weight_layers('input', para_embeddings_op, l2_coef=0.)['weighted_op']
        with tf.variable_scope('elmo_encodings_input', reuse=True):
            elmo_que_input = weight_layers('input', que_embeddings_op, l2_coef=0.)['weighted_op']
        with tf.variable_scope('elmo_encodings_output'):
            elmo_para_output = weight_layers('output', para_embeddings_op, l2_coef=0.)['weighted_op']
        with tf.variable_scope('elmo_encodings_output', reuse=True):
            elmo_que_output = weight_layers('output', que_embeddings_op, l2_coef=0.)['weighted_op']

        with tf.variable_scope("embedding"):
            with tf.device("/cpu:0"):
                ch_emb = tf.reshape(tf.nn.embedding_lookup(
                        self.char_mat, self.para_char), [self.N * self.PL, self.CL, self.dc])
                qh_emb = tf.reshape(tf.nn.embedding_lookup(
                        self.char_mat, self.que_char), [self.N * self.QL, self.CL, self.dc])

            # Bidaf style conv-highway encoder
            ch_emb = conv(ch_emb, self.dc, bias=True, activation=tf.nn.relu,
                          kernel_size=5, name="char_conv", reuse=None)
            qh_emb = conv(qh_emb, self.dc, bias=True, activation=tf.nn.relu,
                          kernel_size=5, name="char_conv", reuse=True)

            ch_emb = tf.reduce_max(ch_emb, axis=1)
            qh_emb = tf.reduce_max(qh_emb, axis=1)

            ch_emb = tf.reshape(ch_emb, [self.N, self.PL, ch_emb.shape[-1]])
            qh_emb = tf.reshape(qh_emb, [self.N, self.QL, qh_emb.shape[-1]])

            with tf.device("/cpu:0"):
                c_emb = tf.nn.dropout(tf.nn.embedding_lookup(self.word_mat, self.para1), 1.0 - self.dropout)
                q_emb = tf.nn.dropout(tf.nn.embedding_lookup(self.word_mat, self.que1), 1.0 - self.dropout)

            para_emb = tf.concat([c_emb, ch_emb], axis=2)
            que_emb = tf.concat([q_emb, qh_emb], axis=2)

        # add elmo
        embedded_para = tf.concat([para_emb, elmo_para_input], -1)
        embedded_que = tf.concat([que_emb, elmo_que_input], -1)
        print("word embedding:", embedded_para.get_shape().as_list(), embedded_que.get_shape().as_list())

        # input encoder
        with tf.variable_scope("para_encoder"):
            para_rep, _ = bi_cudnn_rnn_encoder('gru', self.d, 1, self.dropout,
                                               embedded_para, self.para_len, self.trainable)
        with tf.variable_scope("para_encoder", reuse=True):
            que_rep, _ = bi_cudnn_rnn_encoder('gru', self.d, 1, self.dropout,
                                              embedded_que, self.que_len, self.trainable)
        print("input encoder:", para_rep.get_shape().as_list(), que_rep.get_shape().as_list())

        # add elmo
        para_rep = tf.concat([para_rep, elmo_para_output], -1)
        que_rep = tf.concat([que_rep, elmo_que_output], -1)
        print("contextual word embedding:", para_rep.get_shape().as_list(), que_rep.get_shape().as_list())

        # bi attention
        with tf.variable_scope("bi_attention"):
            joint_mask = tf.to_float(tf.expand_dims(self.para_mask, 2)) * tf.to_float(tf.expand_dims(self.que_mask, 1))
            para_rep = bidaf_attention(para_rep, que_rep, joint_mask, tri_linear_attention)
            para_rep = tf.nn.dropout(para_rep, 1.0 - self.dropout)
            para_proj = tf.layers.dense(inputs=para_rep, units=self.d * 2, activation=tf.nn.relu,
                                        name="self_attn_input_proj")
        print("bi attention:", para_rep.get_shape().as_list())
        # self attneiton
        with tf.variable_scope("self_attention"):
            with tf.variable_scope("input_proj"):
                self_attn_para_input, _ = bi_cudnn_rnn_encoder('gru', self.d, 1, self.dropout,
                                                               para_proj, self.para_len,
                                                               self.trainable)
            diag_mask = 1.0 - tf.diag(tf.ones((self.PL)))
            context_mask = tf.to_float(tf.expand_dims(self.para_mask, 2)) * tf.to_float(
                    tf.expand_dims(self.para_mask, 1))
            self_attn_para = self_attention(self_attn_para_input, context_mask * diag_mask, tri_linear_attention)
            self_attn_para = tf.nn.dropout(self_attn_para, 1.0 - self.dropout)
            self_attn_para = tf.layers.dense(inputs=self_attn_para, units=self.d * 2, activation=tf.nn.relu,
                                             name="self_attn_output_proj")
            self_attn_para += para_proj
        print("self attention:", self_attn_para.get_shape().as_list())
        # model encoder
        with tf.variable_scope("model_encoder"):
            with tf.variable_scope("start_pointer_proj"):
                start_pointer_input, _ = bi_cudnn_rnn_encoder('gru', self.d, 1, self.dropout,
                                                              self_attn_para, self.para_len, self.trainable)
            start_pointer_input = tf.nn.dropout(start_pointer_input, 1.0 - self.dropout)
            logits1 = tf.layers.dense(inputs=start_pointer_input, units=1, use_bias=False,
                                      name="start_pointer")
            logits1 = mask_logits(tf.reshape(logits1, [self.N, -1]), tf.reshape(self.para_mask, [self.N, -1]))

            with tf.variable_scope("end_pointer_proj"):
                end_pointer_input, _ = bi_cudnn_rnn_encoder('gru', self.d, 1, self.dropout,
                                                            tf.concat([self_attn_para, start_pointer_input], axis=-1),
                                                            self.para_len, self.trainable)
            end_pointer_input = tf.nn.dropout(end_pointer_input, 1.0 - self.dropout)
            logits2 = tf.layers.dense(inputs=end_pointer_input, units=1, use_bias=False,
                                      name="end_pointer")
            logits2 = mask_logits(tf.reshape(logits2, [self.N, -1]), tf.reshape(self.para_mask, [self.N, -1]))

        self.probs1 = tf.nn.softmax(logits1)
        self.probs2 = tf.nn.softmax(logits2)
        # get loss
        losses = tf.nn.softmax_cross_entropy_with_logits(logits=logits1, labels=tf.reshape(self.y1, [self.N, -1]))
        losses2 = tf.nn.softmax_cross_entropy_with_logits(logits=logits2, labels=tf.reshape(self.y2, [self.N, -1]))
        self.batch_loss = losses + losses2
        self.loss = tf.reduce_sum(self.batch_loss * tf.to_float(self.labels)) / (tf.reduce_sum(tf.to_float(self.labels)) + 1e-10)

        # get prediction
        outer = tf.matmul(tf.expand_dims(tf.nn.softmax(logits1), axis=2),
                          tf.expand_dims(tf.nn.softmax(logits2), axis=1))
        outer = tf.matrix_band_part(outer, 0, self.AL)
        bprobs, bindex = tf.nn.top_k(tf.reshape(outer, [-1, self.PL * self.PL]), k=self.config.beam_size)
        self.byp1 = bindex // self.PL
        self.byp2 = bindex % self.PL
        self.bprobs = -tf.log(bprobs)

    def add_train_op(self):
        self.opt = tf.train.AdadeltaOptimizer(learning_rate=1.0)
        grads = self.opt.compute_gradients(self.loss)
        gradients, variables = zip(*grads)
        capped_grads, _ = tf.clip_by_global_norm(gradients, self.config.grad_clip)
        self.train_op = self.opt.apply_gradients(zip(capped_grads, variables), global_step=self.global_step)
