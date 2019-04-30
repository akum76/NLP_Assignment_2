import os
import spacy
import numpy as np
import pandas as pd
from nltk.sentiment.util import mark_negation as negation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import en_core_web_sm
import os
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize        
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB 
from sklearn.svm import LinearSVC 
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from scipy import stats

from z_scoring import z_scoring
from feature_generator import feature_generator
from naive_bayes import naive_bayes


nlp=en_core_web_sm.load()
#import lexicon sentiment



#%%
class Classifier:
    """The Classifier"""
    def __init__(self):
        self.z_score=None
        self.z_dict_positive=None
        self.z_dict_negative=None
        self.z_dict_neutral=None
        self.partioned_z_score=None
        self.partioned_z_dict_positive=None
        self.partioned_z_dict_negative=None
        self.partioned_z_dict_neutral=None
        self.naive=None
        self.liklehood=None
        self.partioned_naive=None
        self.partioned_liklehood=None
        self.lemma_dict=None
        self.logistic_clf=None
        datadir="../resource/"
        self.sentiment_lexicon = pd.read_csv(datadir + "sentiment_lexicon.csv",encoding='latin-1')
        
    def tokenizer(self,path):
        # feel free to make a better tokenization/pre-processing
        original_sentences = []
        processed_sentence=[]
        tokenized_sentences=[]
        label=[]
        category=[]
        target_words=[]
        target_words_start=[]
        target_words_end=[]
    
        with open(path,encoding='utf-8') as file:
            for line in file:
                label.append(line.split("\t")[0])
                category.append(line.split("\t")[1])
                target_words.append(line.split("\t")[2])
                original_sentences.append(line.split("\t")[4])
                
        for sentence in original_sentences:
                sentence = nlp(sentence)
                processed_sentence=[]
                for word in sentence:
                    if word.text!='\n':
                        processed_sentence.append(word.text)
                tokenized_sentences.append(processed_sentence)
    
        #Updates Start and End Words
        for x in range(len(original_sentences)):
            target_word_tokens=nlp(target_words[x])
            try:
                target_words_start.append(tokenized_sentences[x].index(target_word_tokens[0].text))
            except:
                temp = original_sentences[x].split(" ")
                y=0
                for z in temp:
                    if target_word_tokens[0].text in z:
                        target_words_start.append(y)
                    y=y+1
                
            try:
                target_words_end.append(max(tokenized_sentences[x].index(target_word_tokens[-1].text),tokenized_sentences[x].index(target_word_tokens[0].text)+3))
            except:
                try:
                    temp = original_sentences[x].split(" ")
                    y=0
                    for z in temp:
                        if target_word_tokens[-1].text in z:
                            target_words_end.append(max(y,tokenized_sentences[x].index(target_word_tokens[0].text)+3))
                        y=y+1
                except:
                    target_words_end.append(len(tokenized_sentences[x]))
    
        return original_sentences, tokenized_sentences, label, category, target_words, target_words_start,target_words_end
    
    def quick_fix_partion(self,start,end):
        for x in range(len(start)):
            if start[x]>end[x]:
                end[x]=start[x]
        return start,end
    
    #%%
    #Make Lemma Dictionnary
    def make_lemma_dict(self,tokenized_sentences):
        temp=str()
        for sentence in tokenized_sentences:
            for word in sentence:
                temp=temp+" "+ word
        
        temp=nlp(temp)
        for word in temp:
            self.lemma_dict[word.text]=word.lemma_
            
            
    def make_lemma_dict_lexicon(self,sentiment_lexicon):
        temp=str()
        for word in sentiment_lexicon.Word.tolist():
            temp=temp+" "+ word
        temp=nlp(temp)
        for word in temp:
            self.lemma_dict[word.text]=word.lemma_        
        
    #%%
    #Get Vocab
    def get_vocab_dictionnary_corpus(self,sentences):
        vocab_dictionnary = dict()
        for sentence in sentences:
            for word in sentence:
                vocab_dictionnary[word]=vocab_dictionnary.get(word,0)+1
            totalwordcount = len(vocab_dictionnary)
        return vocab_dictionnary, totalwordcount
    
    #Get Vocab by category (Naive Bays)
    def get_vocab_dictionnary_classes(self,sentences,label):
        vocab_dictionnary_positive = dict()
        vocab_dictionnary_negative = dict()
        vocab_dictionnary_neutral = dict()
        totalwordcount_positive=0
        totalwordcount_negative=0
        totalwordcount_neutral=0
        for x in range(len(label)):
            if label[x]=="positive":
                for word in sentences[x]:
                    vocab_dictionnary_positive[word]=(vocab_dictionnary_positive.get(word,0)+1)
                totalwordcount_positive = len(sentences[x])+totalwordcount_positive
            elif label[x]=="negative":
                for word in sentences[x]:
                    vocab_dictionnary_negative[word]=(vocab_dictionnary_negative.get(word,0)+1)
                totalwordcount_negative= len(sentences[x])+totalwordcount_negative
            else:
                for word in sentences[x]:
                    vocab_dictionnary_neutral[word]=(vocab_dictionnary_neutral.get(word,0)+1)
                totalwordcount_neutral= len(sentences[x])+totalwordcount_neutral    
        
        return totalwordcount_neutral,totalwordcount_negative,totalwordcount_positive,vocab_dictionnary_neutral,vocab_dictionnary_positive,vocab_dictionnary_negative
    #%%
    def sentence_partion(self,original_sentences,target_words_start,target_words_end):
        sentence_partion_start=list()
        sentence_partion_end=list()
        conjuctions=["but","yet","however","nonetheless","."]
        
        for x in range(len(original_sentences)):
            sentence=original_sentences[x].replace("\n","")
            sentence=nlp(original_sentences[x])
            end_found=False
            start_found=False
            
            for y in range(target_words_end[x]-1,len(sentence)):
                if sentence[y].text in conjuctions:
                    sentence_partion_end.append(y)
                    end_found=True
                    break
            if end_found==False:
                sentence_partion_end.append(len(sentence))
                        
            for y in range(0,target_words_start[x]-1):
                if sentence[y].text in conjuctions:
                    sentence_partion_start.append(y)
                    start_found=True
                    break
            
            if start_found==False:
                sentence_partion_start.append(0)
                
        return sentence_partion_start,sentence_partion_end

    #%%
    def partion_sentences(self,sentence_partion_start,sentence_partion_end,tokenized_sentences):
        partioned_tokenized_sentence=list()
        for x in range(len(tokenized_sentences)):
            partioned_tokenized_sentence.append(tokenized_sentences[x][sentence_partion_start[x]:sentence_partion_end[x]])
        return partioned_tokenized_sentence
        
    #%%
    def make_inverse_distance_matrix(self,tokenized_sentences,target_word_start,target_word_end):
        inverse_distance_matrix=list()
        j=0
        for sentence in tokenized_sentences:
            temp=[]
            i=0
            for word in sentence:
                temp.append(max(min(abs(target_word_start[j]-i),abs(target_word_end[j]-i),),1))                
                i=i+1
            inverse_distance_matrix.append(np.divide(1,np.array(temp)))
            j=j+1
        return inverse_distance_matrix
        
    #%%
    def label_creator(self,label):
        y_final=list()
        
        for text in label:
            if (text=='positive'):
                y_final.append(1)
            elif (text=='negative'):
                y_final.append(-1)
            else:
                y_final.append(0)
    
        return y_final


    
    
    #############################################
    def train(self, trainfile):
        """Trains the classifier model on the training set stored in file trainfile"""
        #tokenizing sentences
        
        
        original_sentences, tokenized_sentences, label, category, target_words,target_words_start,target_words_end = self.tokenizer(trainfile)
        target_words_start,target_words_end = self.quick_fix_partion(target_words_start,target_words_end )
        
        #partions sentences with conjuctions
        sentence_partion_start,sentence_partion_end=self.sentence_partion(original_sentences,target_words_start,target_words_end)
        #partition sentences
        partioned_tokenized_sentences = self.partion_sentences(sentence_partion_start,sentence_partion_end,tokenized_sentences)

        #tokenized sentences with negation
        tokenized_sentences_negation = [negation(doc) for doc in tokenized_sentences]
                                        
        #same for partioned
        partioned_tokenized_sentences_negation = [negation(doc) for doc in partioned_tokenized_sentences]
                                                  
        #Make Lemma Dictionnary 
        self.lemma_dict=dict()
        self.make_lemma_dict(tokenized_sentences_negation)
        self.make_lemma_dict_lexicon(self.sentiment_lexicon)

        #gets vocab dictionnary
        vocab_dictionnary, totalwordcount = self.get_vocab_dictionnary_corpus(tokenized_sentences_negation)
        partioned_vocab_dictionnary, partioned_totalwordcount = self.get_vocab_dictionnary_corpus(partioned_tokenized_sentences_negation)

        #gets positive and negative vocab dictionnary
        totalwordcount_neutral,totalwordcount_negative,totalwordcount_positive,vocab_dictionnary_neutral,vocab_dictionnary_positive,vocab_dictionnary_negative = self.get_vocab_dictionnary_classes(tokenized_sentences_negation,label)
        partioned_totalwordcount_neutral,partioned_totalwordcount_negative,partioned_totalwordcount_positive,partioned_vocab_dictionnary_neutral,partioned_vocab_dictionnary_positive,partioned_vocab_dictionnary_negative = self.get_vocab_dictionnary_classes(partioned_tokenized_sentences_negation,label)

        
        #make z_score matrix
        self.z_score=z_scoring(totalwordcount,totalwordcount_neutral,totalwordcount_negative,totalwordcount_positive,vocab_dictionnary,vocab_dictionnary_neutral,vocab_dictionnary_positive,vocab_dictionnary_negative)
        self.z_score.make_z_matrix()
        self.z_dict_positive=self.z_score.z_dict_positive
        self.z_dict_negative=self.z_score.z_dict_negative
        self.z_dict_neutral=self.z_score.z_dict_neutral
        
        self.partioned_z_score=z_scoring(partioned_totalwordcount,partioned_totalwordcount_neutral,partioned_totalwordcount_negative,partioned_totalwordcount_positive,partioned_vocab_dictionnary,partioned_vocab_dictionnary_neutral,partioned_vocab_dictionnary_positive,partioned_vocab_dictionnary_negative)
        self.partioned_z_score.make_z_matrix()
        self.partioned_z_dict_positive=self.z_score.z_dict_positive
        self.partioned_z_dict_negative=self.z_score.z_dict_negative
        self.partioned_z_dict_neutral=self.z_score.z_dict_neutral
        
        #make inverse distance matrix
        inverse_distance_matrix=self.make_inverse_distance_matrix(tokenized_sentences_negation,target_words_start,target_words_end)
        
        #run naive bayes in order to populate likelehood table
        self.naive=naive_bayes(vocab_dictionnary,vocab_dictionnary_neutral,vocab_dictionnary_positive,vocab_dictionnary_negative,totalwordcount_positive,totalwordcount_negative,totalwordcount_neutral,totalwordcount)
        self.naive.populate_likelehood()
        self.likelehood=self.naive.likelehood
        
        self.partioned_naive=naive_bayes(partioned_vocab_dictionnary,partioned_vocab_dictionnary_neutral,partioned_vocab_dictionnary_positive,partioned_vocab_dictionnary_negative,partioned_totalwordcount_positive,partioned_totalwordcount_negative,partioned_totalwordcount_neutral,partioned_totalwordcount)
        self.partioned_naive.populate_likelehood()
        self.partioned_likelehood=self.partioned_naive.likelehood

        #make feature matrix
        generator=feature_generator(target_words,original_sentences,tokenized_sentences_negation,tokenized_sentences,inverse_distance_matrix,self.likelehood,self.sentiment_lexicon,self.z_dict_positive,self.z_dict_negative,self.z_dict_neutral,self.naive,self.lemma_dict,False)
        generator.make_sentence_length()
        generator.make_sentence_length_unique()
        generator.make_upper_case_punctuation_count()
        generator.lexicon_list()
        generator.net_sentiment()
        feature_matrix=generator.feature_matrix
        
        generator_partioned=feature_generator(target_words,original_sentences,partioned_tokenized_sentences_negation,partioned_tokenized_sentences,inverse_distance_matrix,self.likelehood,self.sentiment_lexicon,self.z_dict_positive,self.z_dict_negative,self.z_dict_neutral,self.naive,self.lemma_dict,True)
        generator_partioned.make_sentence_length()
        generator_partioned.make_sentence_length_unique()
        generator_partioned.make_upper_case_punctuation_count()
        generator_partioned.lexicon_list()
        generator_partioned.net_sentiment()
        feature_matrix_partioned=generator_partioned.feature_matrix
        
        feature_matrix=pd.concat([feature_matrix,feature_matrix_partioned], axis=1, sort=False)

        X=feature_matrix
        Y=self.label_creator(label)

        self.logistic_clf = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial').fit(X, Y)


    def predict(self, datafile):
        """Predicts class labels for the input instances in file 'datafile'
        Returns the list of predicted labels
        """
        ##tokenizes sentences
        dev_original_sentences, dev_tokenized_sentences,dev_label, dev_category, dev_target_words,dev_target_words_start,dev_target_words_end = self.tokenizer(datafile)
        dev_target_words_start,dev_target_words_end  = self.quick_fix_partion(dev_target_words_start,dev_target_words_end )
        
        dev_sentence_partion_start,dev_sentence_partion_end= self.sentence_partion(dev_original_sentences,dev_target_words_start,dev_target_words_end)
        
        partioned_dev_tokenized_sentences = self.partion_sentences(dev_sentence_partion_start,dev_sentence_partion_end,dev_tokenized_sentences)
        
        dev_tokenized_sentences_negation = [negation(doc) for doc in dev_tokenized_sentences]
        
        partioned_dev_tokenized_sentences_negation = [negation(doc) for doc in partioned_dev_tokenized_sentences]
        
        self.make_lemma_dict(dev_tokenized_sentences_negation)

        
        
        
        
        ##Make Inverse_Dev
        inverse_distance_matrix_dev = self.make_inverse_distance_matrix(dev_tokenized_sentences_negation,dev_target_words_start,dev_target_words_end)    
        
        generator_dev=feature_generator(dev_target_words,dev_original_sentences,dev_tokenized_sentences_negation,dev_tokenized_sentences,inverse_distance_matrix_dev,self.likelehood,self.sentiment_lexicon,self.z_dict_positive,self.z_dict_negative,self.z_dict_neutral,self.naive,self.lemma_dict,False)
        generator_dev.make_sentence_length()
        generator_dev.make_sentence_length_unique()
        generator_dev.make_upper_case_punctuation_count()
        generator_dev.lexicon_list()
        generator_dev.net_sentiment()
        feature_matrix_dev=generator_dev.feature_matrix
        
        
        generator_dev_partioned=feature_generator(dev_target_words,dev_original_sentences,partioned_dev_tokenized_sentences_negation,partioned_dev_tokenized_sentences,inverse_distance_matrix_dev,self.likelehood,self.sentiment_lexicon,self.z_dict_positive,self.z_dict_negative,self.z_dict_neutral,self.naive,self.lemma_dict,True)
        generator_dev_partioned.make_sentence_length()
        generator_dev_partioned.make_sentence_length_unique()
        generator_dev_partioned.make_upper_case_punctuation_count()
        generator_dev_partioned.lexicon_list()
        generator_dev_partioned.net_sentiment()
        feature_matrix_dev_partioned=generator_dev_partioned.feature_matrix
        
        feature_matrix_dev=pd.concat([feature_matrix_dev,feature_matrix_dev_partioned], axis=1, sort=False)
        
        
        X_test=feature_matrix_dev

        
        Y_predict_logistic=self.logistic_clf.predict(X_test)
        
        y_final_label=list()
        
        for text in Y_predict_logistic:
            if (text==1):
                y_final_label.append("positive")
            elif (text==-1):
                y_final_label.append("negative")
            else:
                y_final_label.append("neutral")
    
        

        return y_final_label


        

