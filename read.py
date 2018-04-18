import pickle
import json
with open('./output/out_idx.json', 'rb') as f:
	out_idx = pickle.load(f)
with open('./dicts/wordlist.json', 'r') as f:
	wordlist = json.load(f)
	wordlist = ['<EOS>', '<SOS>'] + wordlist
for s in out_idx:
	words = []
	for w in s:
		words.append(wordlist[w])
	sentence = ' '.join(words)
	print(sentence)