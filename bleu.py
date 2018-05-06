import nltk
import argparse
from tqdm import tqdm

def getArgs():
	parse=argparse.ArgumentParser()
	parse.add_argument('--input', '-i', type=str, help='decode output', required=True)
	parse.add_argument('--golden', '-g', type=str, help='golden question', required=True)
	args=parse.parse_args()
	return vars(args)


def main():
	args = getArgs()
	inFile = args['input']
	goldenFile = args['golden']
	decode = []
	with open(inFile, 'r') as fh:
		for line in fh.readlines():
			decode.append(line.strip())
	golden = []
	with open(goldenFile, 'r') as fg:
		for line in fg.readlines():
			golden.append(line.strip())

	sf = nltk.translate.bleu_score.SmoothingFunction()
	weight = (0.25, 0.25, 0.25, 0.25)

	bleu = 0.0
	assert len(decode) == len(golden)
	for i in tqdm(range(len(decode))):
		ques = golden[i].split(' ')
		out_idx = decode[i].split(' ')
		bleu += nltk.translate.bleu(references=[ques], hypothesis=out_idx, smoothing_function=sf.method4, weights=weight)

	bleu /= len(decode)
	print('bleu = %f' % bleu)

if __name__ == '__main__':
	main()
