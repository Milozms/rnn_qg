import linecache
from tqdm import tqdm
import re

def filt_sub():
	files = ['/home/laiyx/30m/fqFiltered2R.txt','/home/laiyx/30m/fqFiltered.txt']
	fbsub2name = {}
	for line in tqdm(linecache.getlines('./dicts/fb_en_title_final.txt')):
		line = line.strip()
		tokens = line.split('\t')
		if len(tokens) > 1:
			name = re.sub('\(.*\)', '', tokens[1]).strip()
			fbsub2name[tokens[0]] = name
	for line in tqdm(linecache.getlines('./dicts/entity_2.txt')):
		line = line.strip()
		tokens = line.split('\t')
		if len(tokens) > 1:
			name = tokens[1]
			fbsub2name[tokens[0]] = name
	exist_entities = set()
	exist_relations = set()
	for line in tqdm(linecache.getlines('/home/zhangms/fb5m/entity2id.txt')):
		line = line.strip()
		tokens = line.split()
		exist_entities.add(tokens[0])
	for line in tqdm(linecache.getlines('/home/zhangms/fb5m/relation2id.txt')):
		line = line.strip()
		tokens = line.split()
		exist_relations.add(tokens[0])
	outf = open('./sq/newdata.txt', 'w')
	cnt = 0
	havesubcnt = 0
	for file in files:
		for line in tqdm(linecache.getlines(file)):
			tokens = line.strip('\n').split('\t')
			pred = tokens[1]
			if pred[0] == '<': # not sq data
				cnt += 1
				split_sub = tokens[0][1:-1].split('/')
				sub = split_sub[-1]
				if sub in fbsub2name:
					# have subject name
					triples = []
					for ent in tokens[0:3]:
						ent = ent[1:-1] # <>
						ent_split = ent.split('/')
						ent = ent_split[-1]
						ent_split = ent.split('.')
						ent = '/' + '/'.join(ent_split)
						triples.append(ent)
					if triples[0] in exist_entities and triples[2] in exist_entities and triples[1] in exist_relations:
						# have entity/relation embedding
						havesubcnt += 1
						triples.append(tokens[3])
						outf.write('\t'.join(triples) + '\n')
	outf.close()
	print('%d triples, %d have subject name and entity/relation embedding' % (cnt, havesubcnt))
	# 30912927 triples, 19475340 have subject name
	# 30912927 triples, 17114026 have subject name and entity/relation embedding


def newdata_triple_format():
	outf = open('./sq/newdata.txt', 'w')
	for line in tqdm(linecache.getlines('./sq/newdata0.txt')):
		tokens = line.strip('\n').split('\t')
		triples = []
		for ent in tokens[0:3]:
			# ent = ent[1:-1] # <>
			# ent_split = ent.split('/')
			# ent = ent_split[-1]
			ent_split = ent.split('.')
			ent = '/'+'/'.join(ent_split)
			triples.append(ent)
		triples.append(tokens[3])
		outf.write('\t'.join(triples)+'\n')
	outf.close()

if __name__ == '__main__':
	filt_sub()
	# newdata_triple_format()
