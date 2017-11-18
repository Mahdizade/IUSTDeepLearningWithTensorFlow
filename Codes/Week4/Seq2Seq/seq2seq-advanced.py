# all codes from https://github.com/ematvey/tensorflow-seq2seq-tutorials
import matplotlib.pyplot as plt
import tensorflow as tf

import helpers

PAD = 0
EOS = 1

vocab_size = 10
input_embedding_size = 20
batch_size = 100

max_batches = 3001
batches_in_epoch = 1000

encoder_hidden_units = 20
decoder_hidden_units = encoder_hidden_units * 2


encoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='encoder_inputs')
encoder_inputs_length = tf.placeholder(shape=(None,), dtype=tf.int32, name='encoder_inputs_length')

decoder_targets = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_targets')


batches = helpers.random_sequences(length_from=3, length_to=8, vocab_lower=2, vocab_upper=10, batch_size=batch_size)


def next_feed():
    next_batch = next(batches)
    encoder_inputs_, encoder_input_lengths_ = helpers.batch(next_batch)
    decoder_targets_, _ = helpers.batch([sequence + [EOS] + [PAD] * 2 for sequence in next_batch])
    return {
        encoder_inputs: encoder_inputs_,
        encoder_inputs_length: encoder_input_lengths_,
        decoder_targets: decoder_targets_,
    }


embeddings = tf.Variable(tf.random_uniform([vocab_size, input_embedding_size], -1.0, 1.0), dtype=tf.float32)

encoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, encoder_inputs)

encoder_cell = tf.contrib.rnn.LSTMCell(encoder_hidden_units)

((encoder_fw_outputs, encoder_bw_outputs),
 (encoder_fw_final_state, encoder_bw_final_state)) = (tf.nn.bidirectional_dynamic_rnn(
    cell_fw=encoder_cell, cell_bw=encoder_cell, inputs=encoder_inputs_embedded,
    sequence_length=encoder_inputs_length, dtype=tf.float32, time_major=True))

encoder_outputs = tf.concat((encoder_fw_outputs, encoder_bw_outputs), 2)

encoder_final_state_c = tf.concat((encoder_fw_final_state.c, encoder_bw_final_state.c), 1)

encoder_final_state_h = tf.concat((encoder_fw_final_state.h, encoder_bw_final_state.h), 1)

encoder_final_state = tf.contrib.rnn.LSTMStateTuple(c=encoder_final_state_c, h=encoder_final_state_h)

decoder_cell = tf.contrib.rnn.LSTMCell(decoder_hidden_units)

encoder_max_time, run_batch_size = tf.unstack(tf.shape(encoder_inputs))

decoder_lengths = encoder_inputs_length + 3

W = tf.Variable(tf.random_uniform([decoder_hidden_units, vocab_size], -1, 1), dtype=tf.float32)
b = tf.Variable(tf.zeros([vocab_size]), dtype=tf.float32)

eos_time_slice = tf.ones([run_batch_size], dtype=tf.int32, name='EOS')
pad_time_slice = tf.zeros([run_batch_size], dtype=tf.int32, name='PAD')

eos_step_embedded = tf.nn.embedding_lookup(embeddings, eos_time_slice)
pad_step_embedded = tf.nn.embedding_lookup(embeddings, pad_time_slice)


def loop_fn_initial():
    # all False at the initial step
    initial_elements_finished = (0 >= decoder_lengths)

    initial_input = eos_step_embedded
    initial_cell_state = encoder_final_state
    initial_cell_output = None
    # we don't need to pass any additional information
    initial_loop_state = None

    return (initial_elements_finished,
            initial_input,
            initial_cell_state,
            initial_cell_output,
            initial_loop_state)


def loop_fn_transition(time, previous_output, previous_state, previous_loop_state):
    def get_next_input():
        output_logits = tf.add(tf.matmul(previous_output, W), b)
        prediction = tf.argmax(output_logits, axis=1)
        next_input = tf.nn.embedding_lookup(embeddings, prediction)
        return next_input

    # this operation produces boolean tensor of [batch_size]
    # defining if corresponding sequence has ended
    elements_finished = (time >= decoder_lengths)

    # -> boolean scalar
    finished = tf.reduce_all(elements_finished)

    inputs = tf.cond(finished, lambda: pad_step_embedded, get_next_input)
    state = previous_state
    output = previous_output
    loop_state = None

    return (elements_finished,
            inputs,
            state,
            output,
            loop_state)


def loop_fn(time, previous_output, previous_state, previous_loop_state):
    if previous_state is None:    # time == 0
        assert previous_output is None
        return loop_fn_initial()
    else:
        return loop_fn_transition(time, previous_output, previous_state, previous_loop_state)

decoder_outputs_ta, decoder_final_state, _ = tf.nn.raw_rnn(decoder_cell, loop_fn)
decoder_outputs = decoder_outputs_ta.stack()

decoder_max_steps, decoder_batch_size, decoder_dim = tf.unstack(tf.shape(decoder_outputs))
decoder_outputs_flat = tf.reshape(decoder_outputs, (-1, decoder_dim))
decoder_logits_flat = tf.add(tf.matmul(decoder_outputs_flat, W), b)
decoder_logits = tf.reshape(decoder_logits_flat, (decoder_max_steps, decoder_batch_size, vocab_size))

decoder_prediction = tf.argmax(decoder_logits, 2)


stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
    labels=tf.one_hot(decoder_targets, depth=vocab_size, dtype=tf.float32), logits=decoder_logits)

loss = tf.reduce_mean(stepwise_cross_entropy)
train_op = tf.train.AdamOptimizer().minimize(loss)

sess = tf.Session()

sess.run(tf.global_variables_initializer())


def main():
    loss_track = []
    try:
        for batch in range(max_batches):
            fd = next_feed()
            _, l = sess.run([train_op, loss], fd)
            loss_track.append(l)

            if batch == 0 or batch % batches_in_epoch == 0:
                print('batch {}'.format(batch))
                print('minibatch loss: {}'.format(sess.run(loss, fd)))
                predict_ = sess.run(decoder_prediction, fd)
                for i, (inp, pred) in enumerate(zip(fd[encoder_inputs].T, predict_.T)):
                    print('sample {}:'.format(i + 1))
                    print('input     > {}'.format(inp))
                    print('predicted > {}'.format(pred))
                    if i >= 2:
                        break
    except KeyboardInterrupt:
        print('training interrupted')

    plt.plot(loss_track)
    print('loss {:.4f} after {} examples (batch_size={})'.format(
        loss_track[-1], len(loss_track)*batch_size, batch_size))
    plt.show()


if __name__ == '__main__':
    main()
