
import tensorflow as tf
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import ops
 
def sequence_loss_by_example_test(logits, targets, weights,
                             average_across_timesteps=True,
                             softmax_loss_function=None, name=None):
  if len(targets) != len(logits) or len(weights) != len(logits):
    raise ValueError("Lengths of logits, weights, and targets must be the same "
                         "%d, %d, %d." % (len(logits), len(weights), len(targets)))
  with ops.name_scope(name, "sequence_loss_by_example",
                        logits + targets + weights):
    #logits: List of 2D Tensors of shape [batch_size x num_decoder_symbols].
    #targets: List of 1D batch-sized int32 Tensors of the same length as logits.
    #weights: List of 1D batch-sized float-Tensors of the same length as logits.
    #return: shape(log_pers) = [batch_size].
    log_perp_list = []
    for logit, target, weight in zip(logits, targets, weights):
      if softmax_loss_function is None:
        # TODO(irving,ebrevdo): This reshape is needed because
        # sequence_loss_by_example is called with scalars sometimes, which
        # violates our general scalar strictness policy.
        target = array_ops.reshape(target, [-1])
        crossent = nn_ops.sparse_softmax_cross_entropy_with_logits(
            labels=target, logits=logit)
      else:
        #calculate the cross entropy between target and logit
        crossent = softmax_loss_function(labels=target,logits=logit)
        #1D tensor
        #shape(crossent) = [batch_size * num_step]
        #shape(weight) = [batch_size * num_step]
        #* is elementwise product
      log_perp_list.append(crossent * weight)
 
    log_perps = math_ops.add_n(log_perp_list)
    #shape(log_perps) = [batch_size * num_step]
    if average_across_timesteps:
      total_size = math_ops.add_n(weights)
      total_size += 1e-12  # Just to avoid division by 0 for all-0 weights.
      log_perps /= total_size
    return log_perps
 
logits_1 = tf.Variable(initial_value=tf.cast([[1,1,1,2],[1,1,1,1],[2,1,3,2]],tf.float32))
logits_2 = tf.Variable(initial_value=tf.cast([[2,3,3,1],[2,3,3,2],[2,2,2,1]],tf.float32))
#batch_size = 3,num_decoder_symbols = 4, time_step = 2
# logits_1 =tf.Variable(shape=(3,4),initializer=tf.random_normal_initializer(mean=0., stddev=1.),dtype=tf.float32,name="l1")
 
# logits_2 = tf.Variable(shape=(3,4),initializer=tf.random_normal_initializer(mean=0., stddev=1.,),dtype=tf.float32,name="l2")
 
targets_1 = tf.constant([2, 1, 3], dtype=tf.int32)
targets_2 = tf.constant([1, 2, 0], dtype=tf.int32)
#w must be 1D是对每个output loss的损失
w_1 = tf.constant([1,0,1], dtype=tf.float32)
w_2 = tf.constant([0,2,3], dtype=tf.float32)
 
AVERAGE_ACROSS_TIMESTEPS = False
AVERAGE_ACROSS_BATCH = False
 
E = tf.contrib.legacy_seq2seq.sequence_loss_by_example([logits_1,logits_2], [targets_1,targets_2], [w_1,w_2],
                                                       average_across_timesteps=AVERAGE_ACROSS_TIMESTEPS)
D = sequence_loss_by_example_test([logits_1,logits_2], [targets_1,targets_2], [w_1,w_2],
                                  average_across_timesteps=AVERAGE_ACROSS_TIMESTEPS)
F = tf.contrib.legacy_seq2seq.sequence_loss([logits_1,logits_2], [targets_1,targets_2], [w_1,w_2],
                  average_across_timesteps=AVERAGE_ACROSS_TIMESTEPS,
                  average_across_batch=AVERAGE_ACROSS_BATCH)
G = tf.reduce_sum(D)
 
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(D.eval(),'\n',E.eval(),'\n',F.eval(),'\n',G.eval())
