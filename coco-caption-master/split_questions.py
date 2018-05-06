import linecache
import re
file = '../sq/annotated_fb_data_test.txt_with_single_placeholder'
with open('../sq/test_questions.txt', 'w') as outf:
	for line in linecache.getlines(file):
		tokens = line.strip().split('\t')
		sub = tokens[0]
		question = tokens[-1]
		words_ = re.split('[^0-9a-zA-Z<>]+', question)
		words = []
		for word in words_:
			if word != '':
				words.append(word.lower())
		words_ = []
		question = ' '.join(words)
		question = question.replace('<placeholder>', sub)
		outf.write(question + '\n')