import tensorflow as tf
import pickle
import numpy as np
from tensorflow.contrib.legacy_seq2seq.python.ops import seq2seq
from tensorflow.python.ops import ctc_ops as ctc

print("Imported")

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

print("Loaded")

train_targets = []

with open("train_words_text", 'r') as f:

    for line in f.readlines():

        original = ' '.join(line.strip().lower().split(' ')).replace('.', '')
        targets = original.replace(' ', '  ')
        targets = targets.split(' ')

        targets = np.hstack([list(x) for x in targets])

        targets = np.asarray([vocab_c_2_id[x] if x in vocab_c_2_id else vocab_c_2_id['U']
                              for x in targets])
    
        train_targets.append(targets)
        
test_targets = []

with open("test_words_text", "r") as f:
    
    for line in f.readlines():
        original = ' '.join(line.strip().lower().split(' ')).replace('.', '')
        targets = original.replace(' ', '  ')
        targets = targets.split(' ')

        targets = np.hstack([list(x) for x in targets])

        targets = np.asarray([vocab_c_2_id[x] if x in vocab_c_2_id else vocab_c_2_id['U']
                              for x in targets])
    
        test_targets.append(targets)

print("Preprocessed")

num_hidden = 128
num_classes = len(vocab_c_2_id) + 1
batch_size = 1
input_size = 123
max_encoder_input_size = 0
max_decoder_input_size = 0
train_ids = []

for i in range(len(train_mfcc)):
    max_encoder_input_size = max(max_encoder_input_size, len(train_mfcc[i]))
    
for i in range(len(test_mfcc)):
    max_encoder_input_size = max(max_encoder_input_size, len(test_mfcc[i]))
    
# for i in range(len(train_words)):
#     train_words[i] = train_words[i].lower()
#     max_decoder_input_size = max(max_decoder_input_size, len(train_words[i]))
    
for i in range(len(train_mfcc)):
    while train_mfcc[i].shape[0] < max_encoder_input_size:
        train_mfcc[i] = np.vstack([train_mfcc[i], [0] * 123])
        
for i in range(len(test_mfcc)):
    while test_mfcc[i].shape[0] < max_encoder_input_size:
        test_mfcc[i] = np.vstack([test_mfcc[i], [0] * 123])
        
# for i in range(len(train_words)):
#     data = []
#     data2 = []
#     for j in range(max_decoder_input_size):
        
#         if j < len(train_words[i]):
#         if j == 0:
#             data = (np.arange(num_classes) == vocab_c_2_id['G']).astype(np.float32)
#             if train_words[i][j] in vocab_c_2_id:
#                 data2 = (np.arange(num_classes) == vocab_c_2_id[train_words[i][j]]).astype(np.float32)
#             else:
#                 data2 = (np.arange(num_classes) == vocab_c_2_id['U']).astype(np.float32)
#         elif j < len(train_words[i]):
#             if train_words[i][j-1] in vocab_c_2_id:
#                 data = np.vstack([data, (np.arange(num_classes) == vocab_c_2_id[train_words[i][j-1]]).astype(np.float32)])
#             else:
#                 data = np.vstack([data, (np.arange(num_classes) == vocab_c_2_id['U']).astype(np.float32)])
#             if train_words[i][j] in vocab_c_2_id:
#                 data2 = np.vstack([data2, (np.arange(num_classes) == vocab_c_2_id[train_words[i][j]]).astype(np.float32)])
#             else:
#                 data2 = np.vstack([data2, (np.arange(num_classes) == vocab_c_2_id['U']).astype(np.float32)])
#         elif j == len(train_words[i]):
#             if train_words[i][j-1] in vocab_c_2_id:
#                 data = np.vstack([data, (np.arange(num_classes) == vocab_c_2_id[train_words[i][j-1]]).astype(np.float32)])
#             else:
#                 data = np.vstack([data, (np.arange(num_classes) == vocab_c_2_id['U']).astype(np.float32)])
#             data2 = np.vstack([data2, [0] * num_classes]) 
#         else:
#             data = np.vstack([data, [0] * num_classes])
#             data2 = np.vstack([data2, [0] * num_classes])
#     train_ids.append(data)
#     train_targets.append(data2)

print("Preprocessed")

for i in range(len(train_targets)):
    max_decoder_input_size = max(max_decoder_input_size, len(train_targets[i]))
    
for i in range(len(test_targets)):
    max_decoder_input_size = max(max_decoder_input_size, len(test_targets[i]))

for i in range(len(train_targets)):
    while len(train_targets[i]) < max_decoder_input_size:
        train_targets[i] = np.append(train_targets[i], vocab_c_2_id['P'])
        
for i in range(len(test_targets)):
    while len(test_targets[i]) < max_decoder_input_size:
        test_targets[i] = np.append(test_targets[i], vocab_c_2_id['P'])
        
        
train_ts = []
for i in range(len(train_targets)):
    data = []
    for j in range(len(train_targets[i])):
        if j == 0:
            data = (np.arange(num_classes) == train_targets[i][j]).astype(np.float32)
        else:
            data = np.vstack([data, (np.arange(num_classes) == train_targets[i][j]).astype(np.float32)])
    train_ts.append(data)
    
test_ts = []
for i in range(len(test_targets)):
    data = []
    for j in range(len(test_targets[i])):
        if j == 0:
            data = (np.arange(num_classes) == test_targets[i][j]).astype(np.float32)
        else:
            data = np.vstack([data, (np.arange(num_classes) == test_targets[i][j]).astype(np.float32)])
    test_ts.append(data)

print("Preprocessed")

weights_attend = tf.Variable(tf.truncated_normal([2 * num_hidden, num_hidden],
                                                   stddev=np.sqrt(2.0 / (2*num_hidden))))

biases_attend = tf.Variable(tf.zeros([num_hidden]))

# weightsClasses = tf.Variable(tf.truncated_normal([num_hidden, num_classes],
#                                                      stddev=np.sqrt(2.0 / num_hidden)))
# biasesClasses = tf.Variable(tf.zeros([num_classes]))


inputs = tf.placeholder(tf.float32, shape = (max_encoder_input_size, batch_size, input_size))
inputrs = tf.reshape(inputs, [-1, input_size])

inputList = tf.split(inputrs, max_encoder_input_size, 0)

tar = tf.placeholder(tf.float32, shape = (max_decoder_input_size, batch_size, num_classes))

tars = []
for j in range(max_decoder_input_size):
    tars.append(tar[j])

fw = tf.contrib.rnn.LSTMCell(num_hidden, state_is_tuple = True)
bw = tf.contrib.rnn.LSTMCell(num_hidden, state_is_tuple = True)

output, _, _ = tf.contrib.rnn.static_bidirectional_rnn(fw, bw, inputList, dtype = tf.float32)

# outH1 = [tf.reduce_sum(tf.multiply(t, weightsOutH1), reduction_indices = 1) + biasesOutH1 for t in outputrs]

decoder_attention_states = [tf.matmul(t, weights_attend) + biases_attend for t in output]
decoder_attention_states = tf.reshape(decoder_attention_states, shape = (batch_size, max_encoder_input_size, num_hidden))

decoder_cell = tf.contrib.rnn.LSTMCell(num_hidden)
decoder_cell_state = decoder_cell.zero_state(batch_size, dtype = tf.float32)

output_decoder, decoder_cell_state = tf.contrib.legacy_seq2seq.attention_decoder(tars, decoder_cell_state, 
                                        decoder_attention_states, decoder_cell)

weights_decoder = tf.Variable(tf.truncated_normal([num_hidden, num_classes],
                                                   stddev=np.sqrt(2.0 / (2*num_hidden))))
biases_decoder = tf.Variable(tf.zeros([num_classes]))

output_decoder = tf.reshape(output_decoder, shape = (max_decoder_input_size, num_hidden))
output_logits = tf.matmul(output_decoder, weights_decoder) + biases_decoder

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = output_logits, labels = tar))

optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

print("Graph created")

num_steps = 100

with tf.Session() as session:
    
    print("Initializing")
    tf.global_variables_initializer().run()
    
    saver = tf.train.Saver()
    
    for step in range(num_steps):
        
        l2 = 0
        for i in range(len(train_mfcc)):
            
            feedDict = {inputs : train_mfcc[i].reshape([max_encoder_input_size, batch_size, input_size]),
                        tar : train_ts[i].reshape([max_decoder_input_size, batch_size, num_classes])}
            
            
            _, l = session.run([optimizer, loss], feed_dict=feedDict)
            
            l2 += l
            
            if i % 500 == 0:
                print("I", i, l2)
                    
        print("Step", step, "Loss", l2)
        saver.save(session, "./Models/bidirectionaldecoder.ckpt")
        
        l2 = 0
        for i in range(len(test_mfcc)):
            
            feedDict = {inputs : test_mfcc[i].reshape([max_encoder_input_size, batch_size, input_size]),
                        tar : test_ts[i].reshape([max_decoder_input_size, batch_size, num_classes])}
            
            
            l = session.run([loss], feed_dict=feedDict)
            
            l2 += l
            
            if i % 500 == 0:
                print("I - test", i, l2)
                    
        print("Test Loss", l2)