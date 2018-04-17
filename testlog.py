import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = logging.FileHandler("./log/log.txt", mode='w')
handler.setLevel(logging.INFO)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
handler.setFormatter(formatter)
console.setFormatter(formatter)
logger.addHandler(handler)
logger.addHandler(console)

for i in range(10):
	logging.info('info %d' % i)