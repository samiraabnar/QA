import numpy as np
import tensorflow as tf


from Model_Base import Model_Base
from EpisodeModule import EpisodeModule

from nn import weight, bias, batch_norm, dropout


class DMN(Model_Base):
  """ Dynamic Memory Networks (March 2016 Version - https://arxiv.org/abs/1603.01417)
      Improved End-To-End version."""

  def build(self):
    params = self.params
    V, d, A = params.embed_size, params.hidden_size, self.words.vocab_size

    # initialize self
    # placeholders
    input = tf.placeholder('int32', shape=[self.params.batch_size, self.params.max_fact_count, self.params.max_sent_size], name='x')  # [num_batch, fact_count, sentence_len]
    question = tf.placeholder('int32', shape=[self.params.batch_size, self.params.max_ques_size], name='q')  # [num_batch, question_len]
    answer = tf.placeholder('int32', shape=[self.params.batch_size], name='y')  # [num_batch] - one word answer
    fact_counts = tf.placeholder('int64', shape=[self.params.batch_size], name='fc')
    input_mask = tf.placeholder('float32', shape=[self.params.batch_size, self.params.max_fact_count, self.params.max_sent_size,self.params.embed_size], name='xm')
    is_training = tf.placeholder(tf.bool)
    self.att = tf.constant(0.)

    # Prepare parameters
    gru = tf.nn.rnn_cell.GRUCell(self.params.hidden_size)
    l = self.positional_encoding()
    embedding = weight('embedding', [self.words.vocab_size, self.params.embed_size], init='uniform', range=3 ** (1 / 2))

    with tf.name_scope('SentenceReader'):
      input_list = tf.unstack(tf.transpose(input))  # L x [F, N]
      input_embed = []
      for facts in input_list:
        facts = tf.unstack(facts)
        embed = tf.stack([tf.nn.embedding_lookup(embedding, w) for w in facts])  # [F, N, V]
        input_embed.append(embed)

      # apply positional encoding
      input_embed = tf.transpose(tf.stack(input_embed), [2, 1, 0, 3])  # [N, F, L, V]
      encoded = l * input_embed * input_mask
      facts = tf.reduce_sum(encoded, 2)  # [N, F, V]

    # dropout time
    facts = dropout(facts, params.keep_prob, is_training)

    with tf.name_scope('InputFusion'):
      # Bidirectional RNN
      with tf.variable_scope('Forward'):
        forward_states, _ = tf.nn.dynamic_rnn(gru, facts, fact_counts, dtype=tf.float32)

      with tf.variable_scope('Backward'):
        facts_reverse = tf.reverse_sequence(facts, fact_counts, 1)
        backward_states, _ = tf.nn.dynamic_rnn(gru, facts_reverse, fact_counts, dtype=tf.float32)

      # Use forward and backward states both
      facts = forward_states + backward_states  # [N, F, d]

    with tf.variable_scope('Question'):
      tf.logging.info(question)
      ques_list = tf.unstack(tf.transpose(question))
      tf.logging.info(ques_list)
      ques_embed = tf.stack([tf.nn.embedding_lookup(embedding, w) for w in ques_list])
      #ques_embed = tf.expand_dims(ques_embed, 0)
      tf.logging.info(ques_embed)
      initial_state = gru.zero_state(self.params.batch_size, dtype=tf.float32)
      _, question_vec = tf.nn.dynamic_rnn(gru, ques_embed,initial_state=initial_state, dtype=tf.float32,time_major=True)

    # Episodic Memory
    with tf.variable_scope('Episodic'):
      episode = EpisodeModule(self.params.hidden_size, question_vec, facts, is_training, self.params.batch_norm)
      memory = tf.identity(question_vec)

      for t in range(params.memory_step):
        with tf.variable_scope('Layer%d' % t) as scope:
          if params.memory_update == 'gru':
            memory = gru(episode.new(memory), memory)[0]
          else:
            # ReLU update
            c = episode.new(memory)
            concated = tf.concat([memory, c, question_vec],1)

            w_t = weight('w_t', [3 * d, d])
            z = tf.matmul(concated, w_t)
            if params.batch_norm:
              z = batch_norm(z, is_training)
            else:
              b_t = bias('b_t', d)
              z = z + b_t
            memory = tf.nn.relu(z)  # [N, d]

          scope.reuse_variables()

    # Regularizations
    if params.batch_norm:
      memory = batch_norm(memory, is_training=is_training)
    memory = dropout(memory, params.keep_prob, is_training)

    with tf.name_scope('Answer'):
      # Answer module : feed-forward version (for it is one word answer)
      w_a = weight('w_a', [d, A], init='xavier')
      logits = tf.matmul(memory, w_a)  # [N, A]

    with tf.name_scope('Loss'):
      # Cross-Entropy loss
      cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=answer)
      loss = tf.reduce_mean(cross_entropy)
      total_loss = loss + params.weight_decay * tf.add_n(tf.get_collection('l2'))

    with tf.variable_scope('Accuracy'):
      # Accuracy
      predicts = tf.cast(tf.argmax(logits, 1), 'int32')
      corrects = tf.equal(predicts, answer)
      num_corrects = tf.reduce_sum(tf.cast(corrects, tf.float32))
      accuracy = tf.reduce_mean(tf.cast(corrects, tf.float32))

    # Training
    optimizer = tf.train.AdamOptimizer(params.learning_rate)
    opt_op = optimizer.minimize(total_loss, global_step=self.global_step)

    # placeholders
    self.x = input
    self.xm = input_mask
    self.q = question
    self.y = answer
    self.fc = fact_counts
    self.is_training = is_training

    # tensors
    self.total_loss = total_loss
    self.num_corrects = num_corrects
    self.accuracy = accuracy
    self.opt_op = opt_op

  def positional_encoding(self):
    D, M, N = self.params.embed_size, self.params.max_sent_size, self.params.batch_size
    encoding = np.zeros([M, D])
    for j in range(M):
      for d in range(D):
        encoding[j, d] = (1 - float(j) / M) - (float(d) / D) * (1 - 2.0 * j / M)

    return encoding

  def preprocess_batch(self, batches):
    """ Make padding and masks last word of sentence. (EOS token)
    :param batches: A tuple (input, question, label, mask)
    :return A tuple (input, question, label, mask)
    """
    params = self.params
    input, question, label = batches
    N, L, Q, F = params.batch_size, params.max_sent_size, params.max_ques_size, params.max_fact_count
    V = params.embed_size

    # make input and question fixed size
    new_input = np.zeros([N, F, L])  # zero padding
    input_masks = np.zeros([N, F, L, V])
    new_question = np.zeros([N, Q])
    new_labels = []
    fact_counts = []

    for n in range(N):
      for i, sentence in enumerate(input[n]):
        sentence_len = len(sentence)
        new_input[n, i, :sentence_len] = [self.words.word_to_index(w) for w in sentence]
        input_masks[n, i, :sentence_len, :] = 1.  # mask words

      fact_counts.append(len(input[n]))

      sentence_len = len(question[n])
      new_question[n, :sentence_len] = [self.words.word_to_index(w) for w in question[n]]

      new_labels.append(self.words.word_to_index(label[n]))

    return new_input, new_question, new_labels, fact_counts, input_masks

  def get_feed_dict(self, batches, is_train):
    input, question, label, fact_counts, mask = self.preprocess_batch(batches)
    return {
      self.x: input,
      self.xm: mask,
      self.q: question,
      self.y: label,
      self.fc: fact_counts,
      self.is_training: is_train
    }
