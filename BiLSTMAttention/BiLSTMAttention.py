import tensorflow as tf

class BiLSTMAttention(object):
    """
    A RNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
      self, embedding_mat, non_static, GRU, sequence_length, 
      hidden_layer_size, vocab_size,
      embedding_size, attention_size , l2_reg_lambda=0.0):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_xpos = tf.placeholder(tf.int32, [None, sequence_length], name="input_xpos")
        self.input_xneg = tf.placeholder(tf.int32, [None, sequence_length], name="input_xneg")
        self.real_len_x = tf.placeholder(tf.int32, [None], name="real_len_x")
        self.real_len_xpos = tf.placeholder(tf.int32, [None], name="real_len_xpos")
        self.real_len_xneg = tf.placeholder(tf.int32, [None], name="real_len_xneg")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.batch_size = tf.placeholder(tf.int32, name="batch_size")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            if non_static:
                W = tf.Variable(embedding_mat, name="W")
            else:
                W = tf.constant(embedding_mat, name="W")
            self.embedded_chars_x = tf.nn.embedding_lookup(W, self.input_x)
            self.embedded_chars_xpos = tf.nn.embedding_lookup(W, self.input_xpos)
            self.embedded_chars_xneg = tf.nn.embedding_lookup(W, self.input_xneg)

        # Create a rnn layer
        with tf.name_scope("forward"), tf.variable_scope("forward"):
            if GRU:
                rnn_fw_cell = tf.contrib.rnn.GRUCell(num_units=hidden_layer_size)
            else:
                rnn_fw_cell = tf.contrib.rnn.LSTMCell(num_units=hidden_layer_size)
            rnn_fw_drop_cell = tf.contrib.rnn.DropoutWrapper(
                rnn_fw_cell, output_keep_prob=self.dropout_keep_prob)
            self._fw_state = rnn_fw_cell.zero_state(self.batch_size, tf.float32)

        with tf.name_scope("backward"), tf.variable_scope("backward"):
            if GRU:
                rnn_bw_cell = tf.contrib.rnn.GRUCell(num_units=hidden_layer_size)
            else:
                rnn_bw_cell = tf.contrib.rnn.LSTMCell(num_units=hidden_layer_size)
            rnn_bw_drop_cell = tf.contrib.rnn.DropoutWrapper(
                rnn_bw_cell, output_keep_prob=self.dropout_keep_prob)
            self._bw_state = rnn_bw_cell.zero_state(self.batch_size, tf.float32)

        with tf.name_scope("biRNN_x"), tf.variable_scope("biRNN"):
            self.rnn_outputs_x, rnn_state_x = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=rnn_fw_cell,
                cell_bw=rnn_bw_cell,
                inputs=self.embedded_chars_x,
                sequence_length=self.real_len_x,
                initial_state_fw=self._fw_state,
                initial_state_bw=self._bw_state
                )
            self.rnn_outputs_x = tf.concat(self.rnn_outputs_x, 2)
        with tf.name_scope("biRNN_xpos"), tf.variable_scope("biRNN", reuse=True):
            self.rnn_outputs_xpos, rnn_state_xpos = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=rnn_fw_cell,
                cell_bw=rnn_bw_cell,
                inputs=self.embedded_chars_xpos,
                sequence_length=self.real_len_xpos,
                initial_state_fw=self._fw_state,
                initial_state_bw=self._bw_state
                )
            self.rnn_outputs_xpos = tf.concat(self.rnn_outputs_xpos, 2)
        with tf.name_scope("biRNN_xneg"), tf.variable_scope("biRNN", reuse=True):
            self.rnn_outputs_xneg, rnn_state_xneg = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=rnn_fw_cell,
                cell_bw=rnn_bw_cell,
                inputs=self.embedded_chars_xneg,
                sequence_length=self.real_len_xneg,
                initial_state_fw=self._fw_state,
                initial_state_bw=self._bw_state
                )
            self.rnn_outputs_xneg = tf.concat(self.rnn_outputs_xneg, 2)

        # An attention model
        with tf.name_scope("attention"):
            # Attention mechanism
            parameters = {
                "sequence_length" : sequence_length,
                "W" : tf.Variable(
                      tf.truncated_normal([self.rnn_outputs_x.shape[2].value, attention_size], stddev=0.1), name="W"),
                "b" : tf.Variable(tf.truncated_normal([attention_size], stddev=0.1), name="b"),
                "u" : tf.Variable(tf.truncated_normal([attention_size], stddev=0.1), name="u"),
            }
            self.attention_outputs_x = self.attention(self.rnn_outputs_x, parameters, "x")
            self.attention_outputs_xpos = self.attention(self.rnn_outputs_xpos, parameters, "xpos")
            self.attention_outputs_xneg = self.attention(self.rnn_outputs_xneg, parameters, "xneg")

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            M = tf.Variable(
                tf.truncated_normal([self.attention_outputs_x.shape[1].value, self.attention_outputs_x.shape[1].value], stddev=0.1), 
                name="M")
            l2_loss += tf.nn.l2_loss(M)
            self.xM = tf.matmul(self.attention_outputs_x, M, name="xM")
            self.xM = tf.reshape(self.xM, [-1, 1, self.xM.shape[1].value])
            self.attention_outputs_xpos = tf.reshape(self.attention_outputs_xpos, 
                [-1, self.attention_outputs_xpos.shape[1].value, 1])
            self.attention_outputs_xneg = tf.reshape(self.attention_outputs_xneg, 
                [-1, self.attention_outputs_xneg.shape[1].value, 1])
            self.x_vs_xpos = tf.matmul(self.xM, self.attention_outputs_xpos, name="x_vs_xpos")
            self.x_vs_xneg = tf.matmul(self.xM, self.attention_outputs_xneg, name="x_vs_xneg")
            self.x_vs_xpos = tf.reshape(self.x_vs_xpos, [-1, 1])
            self.x_vs_xneg = tf.reshape(self.x_vs_xneg, [-1, 1])
            
        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            zero = tf.fill(tf.shape(self.x_vs_xpos), 0.0)
            margin = tf.fill(tf.shape(self.x_vs_xpos), 0.5)  #ori 0.05
            self.losses = tf.maximum(zero, tf.subtract(margin, tf.subtract(self.x_vs_xpos, self.x_vs_xneg)))
            self.loss = tf.reduce_mean(self.losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.greater(self.x_vs_xpos, self.x_vs_xneg, name="correct_predictions")
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

    def attention(self, inputs, parameters, name):
        with tf.name_scope("attention_{}".format(name)):
            att = tf.tanh(
                tf.nn.xw_plus_b(tf.reshape(inputs, [-1, inputs.shape[2].value]), parameters["W"], parameters["b"]),
                name="attention_projection")
            logits = tf.matmul(att, tf.reshape(parameters["u"], [-1, 1]), name="attention_logits")
            attention_weights = tf.nn.softmax(
                tf.reshape(logits, [-1, parameters["sequence_length"]]), dim=1,
                name="attention_weights")

            weighted_rnn_output = tf.multiply(
                inputs, tf.reshape(attention_weights, [-1, parameters["sequence_length"], 1]))
            attention_outputs = tf.reduce_sum(weighted_rnn_output, 1, name="attention_outputs")
            ###################
            # max pooling layer
            #attention_outputs = tf.reduce_max(inputs, 1, name="attention_outputs")
            ###################
            dropout_outputs = tf.nn.dropout(
                attention_outputs, self.dropout_keep_prob, name="attention_outputs_{}".format(name))
        return dropout_outputs
