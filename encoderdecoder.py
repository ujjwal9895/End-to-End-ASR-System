import tensorflow as tf
import pickle
import numpy as np
from tensorflow.contrib.legacy_seq2seq.python.ops import seq2seq


print("Preparing data")
train_mfcc = []
train_words = []
test_mfcc = []
test_words = []
vocab_id_2_c = {}
vocab_c_2_id = {}
x = "PGabcdefghijklmnopqrstuvwxyz0123456789 ,.'?-UE"

with open("train_mfcc", "rb") as f:
    train_mfcc = pickle.load(f)
    
with open("train_output_words", "rb") as f:
    train_words = pickle.load(f)
    
with open("test_mfcc", "rb") as f:
    test_mfcc = pickle.load(f)
    
with open("test_output_words", "rb") as f:
    test_words = pickle.load(f)
    
for i in range(len(x)):
    vocab_c_2_id[x[i]] = i + 1
    vocab_id_2_c[i + 1] = x[i]


num_hidden = 128
num_classes = len(vocab_c_2_id)
batch_size = 1
input_size = 123
max_encoder_input_size = 0
max_decoder_input_size = 0
train_ids = []
train_targets = []

for i in range(len(train_mfcc)):
    max_encoder_input_size = max(max_encoder_input_size, len(train_mfcc[i]))
    
for i in range(len(train_words)):
    train_words[i] = train_words[i].lower()
    max_decoder_input_size = max(max_decoder_input_size, len(train_words[i]))
    
for i in range(len(train_mfcc)):
    while train_mfcc[i].shape[0] < max_encoder_input_size:
        train_mfcc[i] = np.vstack([train_mfcc[i], [0] * 123])
    if i % 1000 == 0:
    	print(i)
        
for i in range(len(train_words)):
    data = []
    data2 = []
    for j in range(max_decoder_input_size):
        if j == 0:
            data = (np.arange(num_classes) == vocab_c_2_id['G']).astype(np.float32)
            if train_words[i][j] in vocab_c_2_id:
                data2 = (np.arange(num_classes) == vocab_c_2_id[train_words[i][j]]).astype(np.float32)
            else:
                data2 = (np.arange(num_classes) == vocab_c_2_id['U']).astype(np.float32)
        elif j < len(train_words[i]):
            if train_words[i][j-1] in vocab_c_2_id:
                data = np.vstack([data, (np.arange(num_classes) == vocab_c_2_id[train_words[i][j-1]]).astype(np.float32)])
            else:
                data = np.vstack([data, (np.arange(num_classes) == vocab_c_2_id['U']).astype(np.float32)])
            if train_words[i][j] in vocab_c_2_id:
                data2 = np.vstack([data2, (np.arange(num_classes) == vocab_c_2_id[train_words[i][j]]).astype(np.float32)])
            else:
                data2 = np.vstack([data2, (np.arange(num_classes) == vocab_c_2_id['U']).astype(np.float32)])
        elif j == len(train_words[i]):
            if train_words[i][j-1] in vocab_c_2_id:
                data = np.vstack([data, (np.arange(num_classes) == vocab_c_2_id[train_words[i][j-1]]).astype(np.float32)])
            else:
                data = np.vstack([data, (np.arange(num_classes) == vocab_c_2_id['U']).astype(np.float32)])
            data2 = np.vstack([data2, [0] * num_classes]) 
        else:
            data = np.vstack([data, [0] * num_classes])
            data2 = np.vstack([data2, [0] * num_classes])
    train_ids.append(data)
    train_targets.append(data2)
    if i % 1000 == 0:
    	print(i)


# encoder_inputs = []
# decoder_inputs = []
# tar = []

# for i in xrange(max_encoder_input_size):
#     encoder_inputs.append(tf.placeholder(tf.float32, shape = [None, input_size], name = "encoder{0}".format(i)))
    
# for i in xrange(max_decoder_input_size):
#     decoder_inputs.append(tf.placeholder(tf.float32, shape = [None, num_classes], name = "decoder{0}".format(i)))
#     tar.append(tf.placeholder(tf.float32, shape = [num_classes]))
    
encoder_inputs = tf.placeholder(tf.float32, shape=(max_encoder_input_size, batch_size, input_size))
decoder_inputs = tf.placeholder(tf.float32, shape=(max_decoder_input_size, batch_size, num_classes))

encoder_i = []
for j in range(max_encoder_input_size):
    encoder_i.append(encoder_inputs[j])
    
decoder_i = []
for j in range(max_decoder_input_size):
    decoder_i.append(decoder_inputs[j])

print("Placeholder alloted")


# encoder_i = tf.split(axis = 0, num_or_size_splits = max_encoder_input_size, value = encoder_i)
# decoder_i = tf.split(axis = 0, num_or_size_splits = max_decoder_input_size, value = decoder_i)

cell = tf.contrib.rnn.LSTMCell(num_hidden)

outputs, state = seq2seq.basic_rnn_seq2seq(encoder_i, decoder_i, cell)

weights = tf.get_variable("weights1", [num_hidden, num_classes], dtype = tf.float32)
biases = tf.get_variable("biases1", [num_classes], dtype = tf.float32)

outs = tf.reshape(outputs, [max_decoder_input_size, num_hidden])
logits = tf.matmul(outs, weights) + biases

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = decoder_inputs))

optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

print("Model prepared")

num_steps = 1000

with tf.Session() as session:
    
    init = tf.global_variables_initializer()
    session.run(init)
    saver = tf.train.Saver()
    for step in range(num_steps):
        
        co = 0
        for i in range(len(train_mfcc)):
            
            inputs = train_mfcc[i].reshape([max_encoder_input_size, batch_size, input_size])
#             inputs = []
#             for j in range(max_encoder_input_size):
#                 inputs.append(train_mfcc[i][j].reshape([batch_size, input_size]))
            
            outputs = train_ids[i].reshape([max_decoder_input_size, batch_size, num_classes])
#             outputs = []
#             for j in range(max_decoder_input_size):
#                 outputs.append(train_ids[i][j].reshape([batch_size, num_classes])
#             t = train_targets[i : i + 5]
            
#             x_list = {key: value for (key, value) in zip(encoder_inputs, inputs)}
#             y_list = {key: value for (key, value) in zip(decoder_inputs, outputs)}
#             z_list = {key: value for (key, value) in zip(tar, t)}

#             _, c, _ = session.run([optimizer, cost, outputs],
#                                   feed_dict = dict(x_list.items() + y_list.items()))
            
            _, c = session.run([optimizer, cost],
                                  feed_dict = {encoder_inputs : inputs, decoder_inputs : outputs})
            co += c
            if i % 1000 == 0:
                print(co)
        print("Step", step, "Loss", co)
        saver.save(session, "./Models/model.ckpt")