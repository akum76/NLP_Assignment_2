import numpy as np
import en_core_web_sm
nlp=en_core_web_sm.load()


class z_scoring:
    def __init__(self,totalwordcount,totalwordcount_neutral,totalwordcount_negative,totalwordcount_positive,vocab_dictionnary,vocab_dictionnary_neutral,vocab_dictionnary_positive,vocab_dictionnary_negative):
        self.vocab_dictionnary=vocab_dictionnary
        self.vocab_dictionnary_positive=vocab_dictionnary_positive
        self.vocab_dictionnary_negative=vocab_dictionnary_negative
        self.vocab_dictionnary_neutral=vocab_dictionnary_neutral
        self.totalwordcount_positive=totalwordcount_positive
        self.totalwordcount_negative=totalwordcount_negative
        self.totalwordcount_neutral=totalwordcount_neutral
        self.totalwordcount=totalwordcount
        self.z_matrix=None
        self.keys=None
        self.z_dict_positive=dict()
        self.z_dict_negative=dict()
        self.z_dict_neutral=dict()
    
    def make_z_matrix(self):
        self.z_matrix=np.zeros([len(self.vocab_dictionnary),3])
        i=0
        self.keys=list(self.vocab_dictionnary.keys())

        for key in self.keys:
            temp=key.lower()
            temp=nlp(temp)[0]            
            if temp.lemma_== "-PRON-" or temp.is_stop or temp.is_punct or temp.like_num:
                self.z_matrix[i][0]=0
                self.z_matrix[i][1]=0
                self.z_matrix[i][2]=0
                
            else:
                a_=self.vocab_dictionnary_positive.get(key,0)
                b_=self.vocab_dictionnary_negative.get(key,0)
                c_=self.vocab_dictionnary_neutral.get(key,0)            
                n_prime_a=self.totalwordcount_positive
                n_prime_b=self.totalwordcount_negative
                n_prime_c=self.totalwordcount_neutral
                p_f=self.vocab_dictionnary[key]/self.totalwordcount                        
                
                self.z_matrix[i][0]=a_-n_prime_a*p_f/(np.sqrt(n_prime_a*p_f*(1-p_f)))
                self.z_matrix[i][1]=b_-n_prime_b*p_f/(np.sqrt(n_prime_b*p_f*(1-p_f)))
                self.z_matrix[i][2]=c_-n_prime_c*p_f/(np.sqrt(n_prime_c*p_f*(1-p_f)))

            
            i=i+1
        i=0
        for x in self.keys:            
            self.z_dict_positive[x]=self.z_matrix[i,0]
            self.z_dict_negative[x]=self.z_matrix[i,1]
            self.z_dict_neutral[x]=self.z_matrix[i,2]
            i=i+1                        
