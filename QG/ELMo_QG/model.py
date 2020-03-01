import tensorflow as tf
from bilm import BidirectionalLanguageModel, weight_layers
import sys
sys.path.append('../..')
from LIB.layers import conv, bilinear_attention, _linear


class QGModel(object):
    def __init__(self, config, word_mat=None, elmo_word_mat=None, label_mat=None, pos_mat=None,
                 ner_mat=None, trainable=True):
        self.config = config
        self.N = config.batch_size if trainable else config.test_batch_size
        self.PL = config.para_limit if trainable else config.test_para_limit
        self.QL = config.ques_limit if trainable else config.test_ques_limit
        self.AL = config.ans_limit if trainable else config.test_ans_limit
        self.d = config.qg_hidden
        self.dw = config.elmo_word_dim
        self.dropout = config.dropout
        self.layer = config.decoder_layers
        self.diverse_beam = config.diverse_beam

        self.global_step = tf.get_variable('global_step', shape=[], dtype=tf.int32,
                                           initializer=tf.constant_initializer(0), trainable=False)
        self.qa_id = tf.placeholder(tf.int32, [self.N], "qa_id")
        self.dropout = tf.placeholder_with_default(0.0, (), name="dropout")
        self.temperature = tf.placeholder_with_default(1.0, (), name="temperature")
        self.diverse_rate = tf.placeholder_with_default(0.0, (), name="diverse_rate")
        self.para = tf.placeholder(tf.int32, [self.N, self.PL], "paragraph")
        self.para_unk = tf.placeholder(tf.int32, [self.N, self.PL + 2], "paragraph_unk")
        self.que = tf.placeholder(tf.int32, [self.N, self.QL + 2], "question")
        self.labels = tf.placeholder(tf.int32, [self.N, self.PL], "labels")
        self.pos_tags = tf.placeholder(tf.int32, [self.N, self.PL], "pos_tags")
        self.ner_tags = tf.placeholder(tf.int32, [self.N, self.PL], "ner_tags")

        # embeddings
        self.word_mat = tf.get_variable("word_mat", initializer=tf.constant(
                elmo_word_mat if config.use_elmo else word_mat, dtype=tf.float32), trainable=config.word_trainable)
        self.label_mat = tf.get_variable("label_mat", initializer=tf.constant(
                label_mat, dtype=tf.float32), trainable=True)
        self.pos_mat = tf.get_variable("pos_mat", initializer=tf.constant(
                pos_mat, dtype=tf.float32), trainable=True)
        self.ner_mat = tf.get_variable("ner_mat", initializer=tf.constant(
                ner_mat, dtype=tf.float32), trainable=True)

        # the enlarged word mat, because each UNK word is represented as (vocabulary size + paragraph position index)
        plus_word_mat = tf.tile(tf.nn.embedding_lookup(self.word_mat, [1]), [self.PL, 1])
        self.plus_word_mat = tf.concat([self.word_mat, plus_word_mat], axis=0)
        self.NV = len(word_mat)
        self.NVP = self.NV + self.PL

        # elmo representation
        bilm = BidirectionalLanguageModel(config.elmo_options_file, config.elmo_weight_file,
                                          use_character_inputs=False, embedding_weight_file=config.embedding_file)
        para_embeddings_op = bilm(self.para_unk)
        self.elmo_para_input = weight_layers('input', para_embeddings_op, l2_coef=0.001)['weighted_op']

        self.para_mask = tf.cast(self.para, tf.bool)
        self.para_len = tf.reduce_sum(tf.cast(self.para_mask, tf.int32), axis=-1)

        # encoder lstm cells
        self.encoder_cells = [tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(self.d),
                                                            input_keep_prob=1.0 - self.dropout) for _ in range(4)]
        self.input_encoder_cells = [tf.nn.rnn_cell.MultiRNNCell(self.encoder_cells[:2]),
                                    tf.nn.rnn_cell.MultiRNNCell(self.encoder_cells[2:])]
        # decoder lstm cells
        self.decoder_cells = [tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(self.d),
                              input_keep_prob=1.0 - self.dropout) for _ in range(self.layer)]
        self.decoder_cell = self.decoder_cells[0] if self.layer == 1 else \
            tf.nn.rnn_cell.MultiRNNCell(self.decoder_cells)

    def add_train_op(self):
        # add train op
        lr = tf.minimum(self.config.ml_learning_rate, 0.001 / tf.log(999.) *
                        tf.log(tf.cast(self.global_step, tf.float32) + 1))
        self.opt = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.8, beta2=0.999, epsilon=1e-7)
        grads = self.opt.compute_gradients(self.loss,
                                           aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)
        gradients, variables = zip(*grads)
        capped_grads, _ = tf.clip_by_global_norm(gradients, self.config.grad_clip)
        self.train_op = self.opt.apply_gradients(zip(capped_grads, variables), global_step=self.global_step)

    def build_graph(self):
        para_emb = self.input_embedding(self.para, self.labels, self.pos_tags, self.ner_tags, self.elmo_para_input)
        para_enc, para_h_end, para_c_end = self.input_encoder(para_emb, self.para_len)
        para_enc_ = self.gated_self_attention(para_enc)
        self.enc = para_enc_
        self.init_h = para_h_end
        self.init_c = para_c_end
        outputs, oups, attn_ws = self.decode(self.que)
        batch_loss = self._compute_loss(outputs, oups, attn_ws)
        self.loss = tf.reduce_mean(batch_loss)
        if self.config.sample:
            self.symbols, self.probs = self.sample()
        else:
            self.symbols, self.probs = self.search(self.config.beam_size)

    def input_embedding(self, words, labels, pos_tags, ner_tags, elmo_rep):
        """input embedding layer: word + label + pos + ner"""
        with tf.variable_scope("Input_Embedding_Layer"):
            sent_word_emb = tf.nn.embedding_lookup(self.plus_word_mat, words)
            sent_label_emb = tf.nn.embedding_lookup(self.label_mat, labels)
            sent_pos_emb = tf.nn.embedding_lookup(self.pos_mat, pos_tags)
            sent_ner_emb = tf.nn.embedding_lookup(self.ner_mat, ner_tags)
            sent_emb = tf.concat([elmo_rep if self.config.use_elmo else sent_word_emb, sent_label_emb,
                                  sent_pos_emb, sent_ner_emb], axis=-1)
            return sent_emb

    def input_encoder(self, emb, length, reuse=None):
        """input encoder layer: 2-layer bi-lstm"""
        with tf.variable_scope("Input_Encoder_Layer", reuse=reuse):
            (h_fw, h_bw), (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(self.input_encoder_cells[0],
                                                                                 self.input_encoder_cells[1], emb,
                                                                                 dtype='float',
                                                                                 sequence_length=length,
                                                                                 scope='input_encoder')
            _, (c_end_fw, h_end_fw) = state_fw
            _, (c_end_bw, h_end_bw) = state_bw
            h = tf.concat([h_fw, h_bw], axis=-1)
            h_end = tf.concat([h_end_fw, h_end_bw], axis=-1)
            c_end = tf.concat([c_end_fw, c_end_bw], axis=-1)
            return h, h_end, c_end

    def gated_self_attention(self, enc):
        """gated self-attention layer"""
        with tf.variable_scope("Gated_Self_Attention_Layer"):
            attn_dim = enc.get_shape().as_list()[-1]
            s, _, _ = bilinear_attention(queries=enc, units=attn_dim, memory=enc, num_heads=1, mask=self.para_mask,
                                         bias=False, return_weights=True)
            f = tf.nn.tanh(conv(tf.concat([enc, s], axis=-1), output_size=attn_dim, bias=False, name="f"))
            g = tf.nn.sigmoid(conv(tf.concat([enc, s], axis=-1), output_size=attn_dim, bias=False, name="g"))
            return g * f + (1 - g) * enc

    def decode(self, que, reuse=None):
        """decoding function used during training, decoder is a 2-layer uni-lstm"""
        with tf.variable_scope("Decoder_Layer", reuse=reuse):
            memory = self.enc
            # init the decoder's state
            h = tf.nn.tanh(_linear(self.init_h, output_size=self.d, bias=False, scope="h_initial"))
            c = tf.nn.tanh(_linear(self.init_c, output_size=self.d, bias=False, scope="c_initial"))
            hh = tf.zeros((self.N, self.d))  # the attention vector from previous step
            state = (c, h) if self.layer == 1 else [(c, h) for _ in range(self.layer)]
            attn_ws = []   # save every step's attention logits
            outputs = []   # save every step's output vectors
            # the ground-truth question
            oups = tf.split(que, [1] * (self.QL + 2), 1)
            for i, inp in enumerate(oups):
                if i > 0:
                    tf.get_variable_scope().reuse_variables()
                # word embedding + previous attention vector
                einp = tf.reshape(tf.nn.embedding_lookup(self.plus_word_mat, inp), [self.N, self.dw])
                cinp = tf.concat([einp, hh], 1)
                # update cell
                h, state = self.decoder_cell(cinp, state)

                # attention, obtain the context vector and attention logits
                attn, _, attn_w = bilinear_attention(tf.expand_dims(h, 1), units=self.d, num_heads=1,
                                                     memory=memory, scope="temporal_attention",
                                                     mask=self.para_mask, bias=False, return_weights=True)
                attn_dim = attn.get_shape().as_list()[-1]
                attn = tf.reshape(attn, [-1, attn_dim])
                attn_w = tf.reshape(attn_w, [-1, self.PL])
                attn_ws.append(attn_w)

                # attention vector
                hh = tf.nn.tanh(_linear(tf.concat([attn, h], 1), output_size=self.d, bias=False, scope="hh"))

                with tf.variable_scope("AttnOutputProjection"):
                    # maxout
                    output = _linear(tf.concat([attn, h], 1), output_size=2 * self.dw, bias=False, scope="maxout")
                    output = tf.reshape(output, [-1, self.dw, 2])
                    output = tf.reduce_max(output, 2)
                    outputs.append(output)

            return outputs, oups, attn_ws

    def search(self, beam_size, prev_probs=None):
        """beam search function used during inference, reuse the parameters defined in decode()"""
        with tf.variable_scope("Decoder_Layer", reuse=True):
            memory = self.enc
            # specify the loop function, either for standard beam search or diverse beam search
            loop_function = self._loop_function_diverse_search if self.diverse_beam else self._loop_function_search
            # init the decoder's state
            h = tf.nn.tanh(_linear(self.init_h, output_size=self.d, bias=False, scope="h_initial"))
            c = tf.nn.tanh(_linear(self.init_c, output_size=self.d, bias=False, scope="c_initial"))
            hh = tf.zeros((self.N, 1, self.d))   # the attention vector from previous step
            state = (c, h) if self.layer == 1 else [(c, h) for _ in range(self.layer)]
            prev, attn_w = None, None   # the output vector and attention logits from previous step
            # the accumulated log probabilities of the beam
            prev_probs = prev_probs if prev_probs is not None else tf.zeros((self.N, 1))
            finished = tf.cast(tf.zeros((self.N, 1)), tf.bool)   # whether </S> is encountered
            symbols = []  # the output words at each step in the beam
            attn_ws = []  # the attention logits at each step in the beam
            # the decoder states at each step in the beam
            hs = [tf.reshape(h, [self.N, 1, self.d])]

            # the ground-truth question, only the start token will be used
            oups = tf.split(self.que, [1] * (self.QL + 2), 1)
            for i, inp in enumerate(oups):
                einp = tf.nn.embedding_lookup(self.plus_word_mat, inp)
                if prev is not None:
                    # from the second step
                    with tf.variable_scope("loop_function", reuse=True):
                        einp, prev_probs, index, prev_symbol, finished = loop_function(beam_size, prev, attn_w,
                                                                                       prev_probs, finished, i)
                        hh = tf.gather_nd(hh, index)   # update prev attention vector
                        state = tuple(tf.gather_nd(s, index) for s in state) if self.layer == 1 else \
                            [tuple(tf.gather_nd(s, index) for s in sta) for sta in state]  # update prev state
                        for j, symbol in enumerate(symbols):
                            symbols[j] = tf.gather_nd(symbol, index)  # update prev symbols
                        symbols.append(prev_symbol)
                        for j, hsi in enumerate(hs):
                            hs[j] = tf.gather_nd(hsi, index)
                # update cell
                state = tuple(tf.reshape(s, [-1, self.d]) for s in state) if self.layer == 1 else \
                    [tuple(tf.reshape(s, [-1, self.d]) for s in sta) for sta in state]
                cinp = tf.concat([einp, hh], -1)
                cinp_dim = cinp.get_shape().as_list()[-1]
                h, state = self.decoder_cell(tf.reshape(cinp, [-1, cinp_dim]), state)
                # compute context vector
                attn, _, attn_w = bilinear_attention(tf.reshape(h, [self.N, -1, self.d]), units=self.d,
                                                     num_heads=1, memory=memory, mask=self.para_mask,
                                                     scope="temporal_attention",
                                                     bias=False, return_weights=True)
                attn_dim = attn.get_shape().as_list()[-1]
                attn = tf.reshape(attn, [-1, attn_dim])
                attn_w = tf.reshape(attn_w, [self.N, -1, self.PL])
                attn_ws.append(attn_w)

                # attention vector
                hh = tf.nn.tanh(_linear(tf.concat([attn, h], -1), output_size=self.d, bias=False, scope="hh"))
                hh = tf.reshape(hh, [self.N, -1, self.d])

                # reshape for next step's indexing convenience
                state = tuple(tf.reshape(s, [self.N, -1, self.d]) for s in state) if self.layer == 1 else \
                    [tuple(tf.reshape(s, [self.N, -1, self.d]) for s in sta) for sta in state]
                hs.append(tf.reshape(h, [self.N, -1, self.d]))

                with tf.variable_scope("AttnOutputProjection"):
                    # maxout
                    output = _linear(tf.concat([attn, h], -1), output_size=2 * self.dw, bias=False, scope="maxout")
                    output = tf.reshape(output, [self.N, -1, self.dw, 2])
                    output = tf.reduce_max(output, -1)

                prev = output

            # process the last symbol
            einp, prev_probs, index, prev_symbol, finished = loop_function(beam_size, prev, attn_w,
                                                                           prev_probs, finished, i)
            for j, symbol in enumerate(symbols):
                symbols[j] = tf.gather_nd(symbol, index)  # update prev symbols
            symbols.append(prev_symbol)

            return symbols, prev_probs

    def sample(self):
        """sampling function used during inference, reuse the parameters defined in decode()"""
        with tf.variable_scope("Decoder_Layer", reuse=True):
            memory = self.enc
            h = tf.nn.tanh(_linear(self.init_h, output_size=self.d, bias=False, scope="h_initial"))
            c = tf.nn.tanh(_linear(self.init_c, output_size=self.d, bias=False, scope="c_initial"))
            hh = tf.zeros((self.N, self.d))
            state = (c, h) if self.layer == 1 else [(c, h) for _ in range(self.layer)]
            prev, attn_w = None, None
            symbols = []
            prev_probs = tf.zeros(self.N)

            # the ground-truth question, only the start token will be used
            oups = tf.split(self.que, [1] * (self.QL + 2), 1)
            for i, inp in enumerate(oups):
                einp = tf.reshape(tf.nn.embedding_lookup(self.plus_word_mat, inp), [self.N, self.dw])
                if prev is not None:
                    with tf.variable_scope("loop_function", reuse=True):
                        einp, prev_symbol, prev_probs = self._loop_function_sample(prev, attn_w, prev_probs, i)
                        symbols.append(prev_symbol)

                cinp = tf.concat([einp, hh], 1)
                h, state = self.decoder_cell(cinp, state)
                # compute context vector
                attn, _, attn_w = bilinear_attention(tf.expand_dims(h, 1), units=self.d, num_heads=1,
                                                     memory=memory, scope="temporal_attention",
                                                     mask=self.para_mask, bias=False, return_weights=True)
                attn_dim = attn.get_shape().as_list()[-1]
                attn = tf.reshape(attn, [-1, attn_dim])
                attn_w = tf.reshape(attn_w, [-1, self.PL])
                # attention vector
                hh = tf.nn.tanh(_linear(tf.concat([attn, h], 1), output_size=self.d, bias=False, scope="hh"))

                with tf.variable_scope("AttnOutputProjection"):
                    # maxout
                    output = _linear(tf.concat([attn, h], 1), output_size=2 * self.dw, bias=False, scope="maxout")
                    output = tf.reshape(output, [-1, self.dw, 2])
                    output = tf.reduce_max(output, 2)

                prev = output

            einp, prev_symbol, prev_probs = self._loop_function_sample(prev, attn_w, prev_probs, i)
            symbols.append(prev_symbol)

            return symbols, tf.expand_dims(prev_probs, 1)

    def _loop_function_sample(self, prev, attn_w, prev_probs, i):
        """sampling loop function"""
        # index helper
        batch_nums_c = tf.tile(tf.expand_dims(tf.range(self.N), 1), [1, self.PL])
        indices_c = tf.stack((batch_nums_c, self.para), axis=2)
        # concat copy logits and generation logits
        logit = tf.matmul(prev, self.word_mat, transpose_b=True)
        full_logit = tf.concat([logit, attn_w], axis=-1)
        dist_g, dist_c = tf.split(tf.nn.softmax(full_logit / self.temperature), [self.NV, self.PL], axis=-1)
        dist_c = tf.scatter_nd(indices_c, dist_c, [self.N, self.NVP])
        plus_dist_g = tf.zeros([self.N, self.PL])
        dist_g = tf.concat([dist_g, plus_dist_g], axis=-1)
        final_dist = dist_g + dist_c

        # sample
        dist = tf.distributions.Categorical(probs=final_dist)
        prev_symbol = dist.sample()
        indices = tf.stack((tf.range(self.N), prev_symbol), axis=1)
        prev_probs += tf.log(tf.clip_by_value(tf.gather_nd(final_dist, indices), 1e-10, 1.0))
        emb_prev = tf.nn.embedding_lookup(self.plus_word_mat, prev_symbol)

        return emb_prev, prev_symbol, prev_probs

    def _loop_function_search(self, beam_size, prev, attn_w, prev_probs, finished, i):
        """standard beam search loop function"""
        dim = 1 if i == 1 else beam_size
        # index helper
        bc = tf.tile(tf.expand_dims(self.para, 1), [1, dim, 1])  # batch_size * beam_size * PL
        batch_nums_c = tf.tile(tf.reshape(tf.range(self.N), [self.N, 1, 1]), [1, dim, self.PL])
        beam_size_c = tf.tile(tf.reshape(tf.range(dim), [1, dim, 1]), [self.N, 1, self.PL])
        # the vocab id of each token in the paragraph
        indices_c = tf.stack((batch_nums_c, beam_size_c, bc), axis=3)

        # generation logits
        logit = tf.matmul(tf.reshape(prev, [-1, self.dw]), self.word_mat, transpose_b=True)
        logit = tf.reshape(logit, [self.N, dim, -1])
        # concat generation logits with attention logits
        full_logit = tf.concat([logit, attn_w], axis=-1)
        # copy and generation probabilities
        dist_g, dist_c = tf.split(tf.nn.softmax(full_logit / self.temperature), [self.NV, self.PL], axis=-1)
        # scatter copy probabilities
        dist_c = tf.scatter_nd(indices_c, dist_c, [self.N, dim, self.NVP])
        # generation probabilities
        plus_dist_g = tf.zeros([self.N, dim, self.PL])
        dist_g = tf.concat([dist_g, plus_dist_g], axis=-1)
        # log final probabilities
        final_dist = tf.log(tf.clip_by_value(dist_g + dist_c, 1e-10, 1.0))

        # beam search
        # mask the finished sentences, to make sure that they stop accumulating probabilities
        finished_mask = tf.concat([tf.zeros((self.N, dim, 1)), tf.zeros((self.N, dim, self.NVP - 1)) - 1e30], axis=-1)
        finished_tmp = tf.expand_dims(tf.to_float(finished), -1)
        final_dist = final_dist * (1. - finished_tmp) + finished_mask * finished_tmp
        # add the previously accumulated probability
        prev_probs = tf.expand_dims(prev_probs, -1)
        prev = final_dist + prev_probs  # batch_size * dim * NVP
        prev = tf.reshape(prev, [self.N, -1])  # batch_size * (dim * NVP)
        # select top-k (k=beam_size) at this step
        probs, prev_symbolb = tf.nn.top_k(prev, beam_size)  # batch_size * beam_size
        # the indexes used to update beam
        index = prev_symbolb // self.NVP
        bindex = tf.tile(tf.expand_dims(tf.range(self.N), -1), [1, beam_size])
        index = tf.stack((bindex, index), axis=2)
        # generated words
        prev_symbol = prev_symbolb % self.NVP
        # check if </S> was sampled, note that '3' should be the vocab id of </S>
        finished = tf.logical_or(tf.gather_nd(finished, index), tf.equal(prev_symbol, 3))
        # embedding_lookup
        emb_prev = tf.nn.embedding_lookup(self.plus_word_mat, prev_symbol)

        return emb_prev, probs, index, prev_symbol, finished

    def _loop_function_diverse_search(self, beam_size, prev, attn_w, prev_probs, finished, i):
        """diverse beam search loop function, method is from the paper: https://arxiv.org/abs/1611.08562"""
        dim = 1 if i == 1 else beam_size
        # index helper
        bc = tf.tile(tf.expand_dims(self.para, 1), [1, dim, 1])  # batch_size * beam_size * PL
        batch_nums_c = tf.tile(tf.reshape(tf.range(self.N), [self.N, 1, 1]), [1, dim, self.PL])
        beam_size_c = tf.tile(tf.reshape(tf.range(dim), [1, dim, 1]), [self.N, 1, self.PL])
        # the vocab id of each token in the paragraph
        indices_c = tf.stack((batch_nums_c, beam_size_c, bc), axis=3)

        # generation logits
        logit = tf.matmul(tf.reshape(prev, [-1, self.dw]), self.word_mat, transpose_b=True)
        logit = tf.reshape(logit, [self.N, dim, -1])
        # concat generation logits with attention logits
        full_logit = tf.concat([logit, attn_w], axis=-1)
        # copy and generation probabilities
        dist_g, dist_c = tf.split(tf.nn.softmax(full_logit), [self.NV, self.PL], axis=-1)
        # scatter copy probabilities
        dist_c = tf.scatter_nd(indices_c, dist_c, [self.N, dim, self.NVP])
        # generation probabilities
        plus_dist_g = tf.zeros([self.N, dim, self.PL])
        dist_g = tf.concat([dist_g, plus_dist_g], axis=-1)
        # log final probabilities
        final_dist = tf.log(tf.clip_by_value(dist_g + dist_c, 1e-10, 1.0))

        # mask the finished sentences, to make sure that they stop accumulating probabilities
        finished_mask = tf.concat([tf.zeros((self.N, dim, 1)), tf.zeros((self.N, dim, self.NVP - 1)) - 1e30], axis=-1)
        finished_tmp = tf.expand_dims(tf.to_float(finished), -1)
        final_dist = final_dist * (1. - finished_tmp) + finished_mask * finished_tmp
        # add the previously accumulated probability
        prev_probs = tf.expand_dims(prev_probs, -1)
        prev = final_dist + prev_probs  # batch_size * dim * NVP

        # diverse beam seach
        # select top-k (k=beam_size) for each item in the beam
        probs, prev_symbolb = tf.nn.top_k(prev, beam_size)  # batch_size * beam_size
        diverse_rank = tf.tile(tf.reshape(tf.range(beam_size), [1, 1, beam_size]), [self.N, dim, 1])
        # penalizing intra-sibling ranking
        dprobs = probs - self.diverse_rate * tf.to_float(diverse_rank)  # batch * dim * beam_size
        dprobs = tf.reshape(dprobs, [self.N, -1])  # batch * (dim * beam_size)
        # select the final top-k (k=beam_size)
        dprobs, dindex = tf.nn.top_k(dprobs, beam_size)
        bindex = tf.tile(tf.expand_dims(tf.range(self.N), -1), [1, beam_size])
        # the indexes used to update beam
        index = tf.stack((bindex, dindex // beam_size), axis=2)
        cindex = tf.stack((bindex, dindex // beam_size, dindex % beam_size), axis=2)
        prev_symbol = tf.gather_nd(prev_symbolb, cindex)
        probs = tf.gather_nd(probs, cindex)
        # check if </S> was sampled, note that '3' should be the vocab id of </S>
        finished = tf.logical_or(tf.gather_nd(finished, index), tf.equal(prev_symbol, 3))
        # embedding_lookup
        emb_prev = tf.nn.embedding_lookup(self.plus_word_mat, prev_symbol)

        return emb_prev, probs, index, prev_symbol, finished

    def _compute_loss(self, ouputs, oups, attn_ws):
        """compute the cross entropy loss at each step"""
        # index helper
        batch_nums_c = tf.tile(tf.expand_dims(tf.range(self.N), 1), [1, self.PL])
        indices_c = tf.stack((batch_nums_c, self.para), axis=2)   # the vocab id of each token in the paragraph
        batch_nums = tf.expand_dims(tf.range(self.N), 1)

        weights = []   # either it is a padding position or not
        crossents = []  # the cross entropy at each step
        for output, oup, attn_w in zip(ouputs[:-1], oups[1:], attn_ws[:-1]):
            # concat copy logits and generation logits
            logit = tf.matmul(output, self.word_mat, transpose_b=True)   # generation logits
            full_logit = tf.concat([logit, attn_w], axis=-1)     # concat generation logits with attention logits
            # generation and copy probabilities
            dist_g, dist_c = tf.split(tf.nn.softmax(full_logit), [self.NV, self.PL], axis=-1)
            # scatter the copy probabilities
            dist_c = tf.scatter_nd(indices_c, dist_c, [self.N, self.NVP])
            # generation probabilities
            plus_dist_g = tf.zeros([self.N, self.PL])
            dist_g = tf.concat([dist_g, plus_dist_g], axis=-1)
            # final probabilities
            final_dist = dist_g + dist_c
            # get loss
            indices = tf.concat((batch_nums, oup), axis=1)
            gold_probs = tf.gather_nd(final_dist, indices)
            crossent = -tf.log(tf.clip_by_value(gold_probs, 1e-10, 1.0))
            # mask padding positions
            target = tf.reshape(oup, [-1])
            weight = tf.cast(tf.cast(target, tf.bool), tf.float32)
            weights.append(weight)
            # mask the loss on padding positions
            crossents.append(crossent * weight)
        # averaged loss over step
        log_perps = tf.add_n(crossents) / (tf.add_n(weights) + 1e-12)
        return log_perps


class QGRLModel(QGModel):
    def __init__(self, config, word_mat=None, elmo_word_mat=None, label_mat=None, pos_mat=None,
                 ner_mat=None, trainable=True):
        QGModel.__init__(self, config, word_mat=word_mat, elmo_word_mat=elmo_word_mat, label_mat=label_mat,
                 pos_mat=pos_mat, ner_mat=ner_mat, trainable=trainable)

        self.reward = tf.placeholder_with_default(tf.ones([self.N]), (self.N,), name="reward")
        self.sampled_que = tf.placeholder_with_default(tf.zeros([self.N, self.QL + 2], dtype=tf.int32),
                                                       (self.N, self.QL + 2), name="sampled_question")
        self.lamda = tf.placeholder_with_default(config.mixing_ratio, (), name="mixing_ratio")

    def build_graph(self):
        para_emb = self.input_embedding(self.para, self.labels, self.pos_tags, self.ner_tags, self.elmo_para_input)
        para_enc, para_h_end, para_c_end = self.input_encoder(para_emb, self.para_len)
        para_enc_ = self.gated_self_attention(para_enc)
        self.enc = para_enc_
        self.init_h = para_h_end
        self.init_c = para_c_end
        # ml
        outputs, oups, attn_ws = self.decode(self.que)
        batch_loss_ml = self._compute_loss(outputs, oups, attn_ws)
        self.loss_ml = tf.reduce_mean(batch_loss_ml)
        # rl
        outputs, oups, attn_ws = self.decode(self.sampled_que, reuse=True)
        batch_loss_rl = self._compute_loss(outputs, oups, attn_ws)
        self.loss_rl = tf.reduce_mean(batch_loss_rl * self.reward)
        self.loss = (1 - self.lamda) * self.loss_ml + self.lamda * self.loss_rl
        self.symbols, self.probs = self.search(1)  # baseline
        self.symbols_rl, self.probs_rl = self.sample()

    def add_train_op(self):
        # add train op
        lr = self.config.rl_learning_rate
        self.opt = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.8, beta2=0.999, epsilon=1e-7)
        grads = self.opt.compute_gradients(self.loss,
                                           aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)
        gradients, variables = zip(*grads)
        capped_grads, _ = tf.clip_by_global_norm(gradients, self.config.grad_clip)
        self.train_op = self.opt.apply_gradients(zip(capped_grads, variables), global_step=self.global_step)