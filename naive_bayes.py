


#Naive Bayes classifier
class naive_bayes():
    def __init__(self,vocab_dictionnary,vocab_dictionnary_neutral,vocab_dictionnary_positive,vocab_dictionnary_negative,totalwordcount_positive,totalwordcount_negative,totalwordcount_neutral,totalwordcount):
        self.prior=list()
        self.likelehood=dict()
        self.vocab_dictionnary=vocab_dictionnary
        self.vocab_dictionnary_positive=vocab_dictionnary_positive
        self.vocab_dictionnary_negative=vocab_dictionnary_negative
        self.vocab_dictionnary_neutral=vocab_dictionnary_neutral
        self.totalwordcount_positive=totalwordcount_positive
        self.totalwordcount_negative=totalwordcount_negative
        self.totalwordcount_neutral=totalwordcount_neutral
        self.totalwordcount=totalwordcount
    
    def populate_likelehood(self):
        for key in self.vocab_dictionnary:
            prob_positive=(self.vocab_dictionnary_positive.get(key,0)+1)/(self.totalwordcount_positive+self.totalwordcount)
            prob_negative=(self.vocab_dictionnary_negative.get(key,0)+1)/(self.totalwordcount_negative+self.totalwordcount)            
            prob_neutral=(self.vocab_dictionnary_neutral.get(key,0)+1)/(self.totalwordcount_neutral+self.totalwordcount) 
            self.likelehood[key]=[prob_positive,prob_negative,prob_neutral]
        
        self.prior=[self.totalwordcount_positive/self.totalwordcount,self.totalwordcount_negative/self.totalwordcount,self.totalwordcount_neutral/self.totalwordcount]
    
    def test(self,sentence):
        prob_positive=1
        prob_negative=1
        prob_neutral=1
        for word in sentence:
            #for new words
            if self.likelehood.get(word,-1)==-1:
                self.likelehood[word]=[(1/(2*self.totalwordcount_positive)),(1/(2*self.totalwordcount_negative)),(1/(2*self.totalwordcount_neutral))]
                           
            prob_positive=self.likelehood[word][0]*prob_positive
            prob_negative=self.likelehood[word][1]*prob_negative
            prob_neutral=self.likelehood[word][2]*prob_neutral

        prob_positive=prob_positive*self.prior[0]
        prob_negative=prob_negative*self.prior[1]
        prob_neutral=prob_neutral*self.prior[2]
        
        max_prob= max(prob_positive,prob_negative,prob_neutral)
        
        if max_prob==prob_positive:
            return "positive"
        elif max_prob==prob_negative:
            return "negative"
        else:
            return "neutral"

    def naive_features(self,sentence):
        prob_positive=1
        prob_negative=1
        prob_neutral=1
        for word in sentence:
            #for new words
            if self.likelehood.get(word,-1)==-1:
                self.likelehood[word]=[(1/(2*self.totalwordcount_positive)),(1/(2*self.totalwordcount_negative)),(1/(2*self.totalwordcount_neutral))]
                           
            prob_positive=self.likelehood[word][0]*prob_positive
            prob_negative=self.likelehood[word][1]*prob_negative
            prob_neutral=self.likelehood[word][2]*prob_neutral

        prob_positive=prob_positive*self.prior[0]
        prob_negative=prob_negative*self.prior[1]
        prob_neutral=prob_neutral*self.prior[2]
        
        sum_probs=prob_positive+prob_negative+prob_neutral
        
        return [prob_positive/sum_probs, prob_negative/sum_probs, prob_neutral/sum_probs]
