import linecache
import re
datasets = ['valid', 'test']
for dset in datasets:
	file = '../sq/annotated_fb_data_'+dset+'.txt_with_single_placeholder'
	with open('../sq/'+dset+'_questions.txt', 'w') as outf:
		for line in linecache.getlines(file):
			tokens = line.strip().split('\t')
			sub = tokens[0]
			triple = tokens[1:4]
			question = tokens[-1]
			words_ = re.split('[^0-9a-zA-Z<>]+', question)
			words = []
			for word in words_:
				if word != '':
					words.append(word.lower())
			words_ = []
			question = ' '.join(words)
			question = question.replace('<placeholder>', sub)
			# outf.write('\t'.join(triple) + '\t' + question + '\n')
			outf.write(question + '\n')