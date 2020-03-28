import tensorflow as tf


def softmax_cross_entrophy_loss(logits, targets):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits,
        labels=targets,
        name="cross_entropy_per_example"
    )
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name="cross_entropy")
    return cross_entropy_mean


def softmax_seq_loss_by_example(logits, targets, batch_size, seq_length):
    loss = tf.nn.seq2seq.sequence_loss_by_example(
        [logits],
        [targets],
        [tf.ones([batch_size * seq_length])])
    seq_cross_entropy_mean = tf.reduce_sum(loss) / batch_size
    return seq_cross_entropy_mean
