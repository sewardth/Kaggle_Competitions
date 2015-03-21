import sys
from operator import itemgetter

class text_source():
	"""docstring for text_source"""
	def __init__(self, path):
		self.path = path
		f = open(path)
		self.text = str(f.readlines())
		f.close()

	def split_text(self):
		self.text = self.text.lower()
		remove_list = ['.','?','!',',','"', "\n',","'\n'","'",'[',']',"\\n",'\\xef\\xbb\\xbfno','(',')']
		bad_words =['about', 'across','against','along','around','at','behind','beside','besides','by',
					   'despite','down','during','for','from','in','inside','into','near','of','off',
					   'on','onto','over','through','to','toward','with','within','without','i','you',
					   'he','me','her','him','my','mine','her','hers','his','myself','himself','herself',
					   'anything','everything','anyone','everyone','ones','such','it','we','they','us','them',
					   'our','ours','their','theirs','itself','ourselves','themselves','something','nothing','someone',
					   'the','some','this','that','every','all','both','one','first','other','next','many','much',
					   'more','most','several','no','a','an','any','each','no','half','twice','two','second','another',
					   'last','few','little','less','least','own','and','but','after','when','as','because','if',
					   'what','where','which','how','than','or','so','before','since','while','although','though',
					   'who','whose','can','may','will','shall','could','be','do','have','might','would','should',
					   'must','here','there','today','tomorrow','now','then','always','never','sometimes','usually',
					   'often','therefore','however','besides','moreover','though','otherwise','else','instead','anyway',
					   'incidentally','meanwhile','is','also','are','been','was','were','not','used','generally',
					   'known','has','see','its','had','apple','apples','apple\\s','\\xc3\\xa2\\xe2\\x82\\xac\\xe2\\x80\\x9c',
					   '\\xc3\\x8e\\xc2\\xbcg','uses','=','m)','-','until','needed']
		for x in remove_list:
			self.text = self.text.replace(x,'')
		self.text = self.text.strip("\n',")
		self.text = self.text.strip("'\n'")
		self.text = self.text.lower()
		text_list = self.text.split()

		new_list = [x for x in text_list if x not in bad_words]

		return new_list

	def __str__(self):
		return str(self.text)

		

class bag():
	def __init__(self, words):
		self.model = text_source(words).split_text()
		self.data = {}
		self.builder()

	def builder(self):
		for x in self.model:
			if x not in self.data:
				self.data[x] = self.model.count(x)

	def top_list(self, numbers=1300):
		records = self.data.items()
		new = sorted(records, key=itemgetter(1) , reverse=True)
		data =[]
		x = 0
		while x < numbers:
			data.append(new[x][0])
			x+=1
		return data





def score(word_list, list_type):
	score = 0
	for x in word_list:
		if x in list_type[0:30]:
			score = score + 1.5
		elif x in list_type[29:50]:
			score = score + 1.25
		elif x in list_type[49:50]:
			score = score + 1.0
		elif x in list_type[49:100]:
			score = score + .90
		elif x in list_type[99:150]:
			score = score + .80
		elif x in list_type[149:200]:
			score = score + .70
		elif x in list_type[199:300]:
			score = score + .60
		elif x in list_type[299:400]:
			score = score + .50
		elif x in list_type[399:500]:
			score = score + .40
		elif x in list_type[499:]:
			score = score + .25
		else:
			score = score + 0
	return score


def main():
	company_file = bag('apple-computers.txt')
	fruit_file = bag('apple-fruit.txt')
	company_bag = set(company_file.top_list())
	fruit_bag = set(fruit_file.top_list())
	company = list(company_bag.difference(fruit_bag))
	fruit = list(fruit_bag.difference(company_bag))
	for line in sys.stdin:
         remove_list = ['.','?','!',',','"', "\n',","'\n'","'",'[',']',"\\n",'\\xef\\xbb\\xbfno','(',')']
         for x in remove_list:
             line = line.replace(x,'')
         line = line.lower()
         line = line.strip()
         comp = line.split()
         fruit_match = score(comp, fruit)
         company_match = score(comp, company)
        
         if len(comp)<2:
            pass
         else:
            if fruit_match >= company_match:
              print 'fruit'
            else:
              print 'computer-company'
           
main()