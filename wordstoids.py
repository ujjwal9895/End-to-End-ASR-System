import pickle

with open("train_output_words", "rb") as fp:
	train_dataset_output = pickle.load(fp)

vocab = {}
vocabid = {}
count = 3

for line in train_dataset_output:
	for word in line:
		if word not in vocab:
			vocab[word] = count
			vocabid[count] = word
			count += 1

train_dataset_words = list()

for line in train_dataset_output:
	train_dataset_words.append([vocab[word] for word in line])
	print(len(train_dataset_words))

with open("train_output_ids", "wb") as fp2:
	pickle.dump(train_dataset_words, fp2, protocol = 2)

with open("vocab_id_2_word", "wb") as f:
	pickle.dump(vocabid, f, protocol = 2)

with open("vocab_word_2_id", "wb") as f:
	pickle.dump(vocab, f, protocol = 2)
"""
with open("train_output_ids", "rb") as fp2:
	train = pickle.load(fp2)
print(train)
"""