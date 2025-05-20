import pandas as pd
import re
from gensim.models import Doc2Vec

class InfoHscode(object):
    
    def __init__(self, pickle_path = './data/hs6_commentary.pickle', lang = 'eng'):
        self.pickle_path = pickle_path
        self.lang = lang
        self.df = pd.read_pickle(self.pickle_path)
        
    def hscode_info(self, hscode):
        hs4 = hscode[:4]
        hs5 = hscode[:5]
        hs4_info = ''
        hs5_info = ''
        hs6_info = ''
        
        try:
            hs4_info = self.df[self.df['hs'] == hs4].iloc[0][self.lang]
        except:
            pass

        try:
            hs5_info = self.df[self.df['hs'] == hs5].iloc[0][self.lang]
        except:
            pass
            
        try:
            hs6_info = self.df[self.df['hs'] == hscode].iloc[0][self.lang]
        except:
            pass
            
            
        return [hs4_info, hs5_info, hs6_info]


class HscodePredict(InfoHscode):
    
    def __init__(self, model_path):
        super().__init__()
        self.model_path = model_path
        self.model = Doc2Vec.load(self.model_path)
        self.stop_words = {'of', 'or', 'and', 'for', 'than', 'the', 'in', 'with', 'to', 'but', 'by', 'whether',
                           'on', 'its', 'an', 'their', 'at', 'this', 'which', 'from', 'as', 'be', 'is'}
        self.alphabet_pattern = re.compile(r'[^a-zA-Z]')
        self.alphabet_number_pattern = re.compile(r'[^a-zA-Z0-9]')
        self.remove_pattern = re.compile(r'[\;\,\)\(\[\]\:]')

    def refine_text_func(self, text):
        text = text.lower()
        text = ' '.join([w for w in text.split() if w not in self.stop_words])
        alphabet = re.sub(self.alphabet_pattern, ' ', text)
        alphabet_number = re.sub(self.alphabet_number_pattern, ' ', text)
        remove = re.sub(self.remove_pattern, ' ', text)
        result = ' '.join([text, alphabet, alphabet_number, remove])
        return result
    
    def create_ngram_data(self, text, ngram_value=2):
        text_list = text.split()
        ngram_list = list(zip(*[text_list[i:] for i in range(ngram_value)]))
        result = []
        for n_data in ngram_list:
            result.append('_'.join(n_data))
        return ' '.join(result)

    def make_refine_data(self, text):
        refine_text = self.refine_text_func(text)
        ngram_text = self.create_ngram_data(refine_text)
        return ' '.join([refine_text, ngram_text])
    
    def predict(self, text, topn=3):
        refined_text = self.make_refine_data(text)
        input_vector = self.model.infer_vector(refined_text.split(), alpha=0.025, min_alpha=0.001, steps=20)
        similarities = self.model.docvecs.most_similar([input_vector], topn=topn)
        simil_result = [(simil[0], round(simil[1], 4)) for simil in similarities]
        return simil_result
    
    def predict_add_info(self, text, topn=3):
        simil_result = self.predict(text, topn=topn)
        result = []
        for simil in simil_result:
            this_data = {'predict': simil[0],
                         'score': simil[1],
                         'hs_info': super().hscode_info(simil[0])}
            result.append(this_data)
        return result