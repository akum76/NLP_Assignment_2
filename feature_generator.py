import pandas as pd
import numpy as np
import en_core_web_sm
nlp=en_core_web_sm.load()




class feature_generator:
    def __init__(self,target_words,original_sentences,tokenized_negation,tokenized,inverse_distance_matrix,likelehood,sentiment_lexicon,z_dict_positive,z_dict_negative,z_dict_neutral,naive,lemma_dict,partioned_boolean=False):
        self.target_words=target_words
        self.original_sentences=original_sentences
        self.partioned_boolean=partioned_boolean
        self.sentiment_lexicon=sentiment_lexicon
        self.tokenized_negation=tokenized_negation
        self.df_negation=pd.DataFrame(tokenized_negation)
        self.feature_matrix=pd.DataFrame()
        self.tokenized=tokenized
        self.pol_list=list()
        self.inverse_distance_matrix=inverse_distance_matrix
        self.likelehood=likelehood
        self.sentiment_lexicon=sentiment_lexicon
        self.z_dict_positive=z_dict_positive
        self.z_dict_negative=z_dict_negative
        self.z_dict_neutral=z_dict_neutral
        self.naive=naive
        self.lemma_dict=lemma_dict
        
        
        
    def make_sentence_length(self):
        self.feature_matrix["sentence_length"+str(self.partioned_boolean)]=self.df_negation.count(axis=1)
        
    def make_sentence_length_unique(self):
        self.feature_matrix["sentence_length_unique"+str(self.partioned_boolean)]=self.df_negation.nunique(axis=1)
                
    def make_upper_case_punctuation_count(self):
        exclam_mark_list=list()
        question_mark_list=list()
        upper_list=list()
        exclam_mark_count=0
        question_mark_count=0
        upper_count=0
        for sentence in self.tokenized_negation:
            upper_tag=0 #only more than one upper letter in a word tags this to 1
            for word in sentence:                
                for character in word:
                    if (character.upper()==character and character.isalpha()==True):
                        upper_tag=upper_tag+1                
                    
                    if character=="?":
                        question_mark_count = question_mark_count  + 1
                            
                    if character=="!":
                        exclam_mark_count = exclam_mark_count  + 1
                        
                if upper_tag>=2:
                    upper_count=upper_count+1
                upper_tag=0 # resets it, for the next word
            
            exclam_mark_list.append(exclam_mark_count)
            exclam_mark_count=0
            
            question_mark_list.append(question_mark_count)
            question_mark_count=0
            
            upper_list.append(upper_count)
            upper_count=0
                            
        self.feature_matrix["exclam_mark_count"+str(self.partioned_boolean)]=exclam_mark_list
        self.feature_matrix["question_mark_count"+str(self.partioned_boolean)]=question_mark_list
        self.feature_matrix["upper_count"+str(self.partioned_boolean)]=upper_list
        

    def lexicon_list(self):
        self.sentiment_lexicon=np.array(self.sentiment_lexicon)
        sentiment_dict=dict()
        for x in range(len(self.sentiment_lexicon)):
            sentiment_dict[self.lemma_dict.get(self.sentiment_lexicon[x][0],0)]=self.sentiment_lexicon[x][1]
            
        temp_list=list()    
        for sentence in self.tokenized_negation:
               for w in sentence:
                   a=self.lemma_dict.get(w,0)
                   temp_list.append(sentiment_dict.get(a,0))
               self.pol_list.append(temp_list)
               temp_list=list()

    def pos_words (self,sentence, token, ptag,sentiment_lexicon):
        character=str()
        adj_feat_pos=0
        adj_feat_neg=0
        adj_feat_neut=0
        sentences = [sent for sent in sentence.sents if token in sent.string]
        sentiment_lexicon=np.array(sentiment_lexicon)     
        pwrds = []
        for sent in sentences:
            for word in sent:
                if character in word.string: 
                       pwrds.extend([child.string.strip() for child in word.children
                                                          if child.pos_ == ptag] )
            for pwrd in pwrds:
                for index in range(len(sentiment_lexicon)):
                    if (pwrd==sentiment_lexicon[index][0]):
                        if sentiment_lexicon[index][1]==1:
                            adj_feat_pos=adj_feat_pos+1
                                
                        elif sentiment_lexicon[index][1]==-1:
                            adj_feat_neg=adj_feat_neg-1
                        else:
                            adj_feat_neut=adj_feat_neut
        if (len(pwrds)==0):
            return 0
        else:
            return ((adj_feat_pos+adj_feat_neg+adj_feat_neut)/len(pwrds))



    def net_sentiment(self):

            net_positive=list()
            net_negative=list()
            net_neutral=list()
                
            net_positive_normalized=list()
            net_negative_normalized=list()
            net_neutral_normalized=list()
            
            net_positive_distance=list()
            net_negative_distance=list()
            net_neutral_distance=list()
            
            net_positive_likelehood=list()
            net_negative_likelehood=list()
            net_neutral_likelehood=list()
                        
            z_positive=list()
            z_negative=list()
            z_neutral=list()
            
            z_positive_distance=list()
            z_negative_distance=list()
            z_neutral_distance=list()

            naive_prob_positive=list()
            naive_prob_negative=list()
            naive_prob_neutral=list()

            final_score_adj=list()
            
            final_score_adv=list()

                                                
            for x in range(len(self.tokenized_negation)):
                
                if self.partioned_boolean == False:
                
                    target_word_applied=self.target_words[x]
                    adj_overall=self.pos_words(nlp(self.original_sentences[x]), target_word_applied, 'ADJ',self.sentiment_lexicon)
                    final_score_adj.append(adj_overall)
                
                    adv_overall=self.pos_words(nlp(self.original_sentences[x]), target_word_applied, 'ADV',self.sentiment_lexicon)
                    final_score_adv.append(adv_overall)
                                
                pol_sentence =self.pol_list[x]
                distance=self.inverse_distance_matrix[x]
                neg_sentence=self.tokenized_negation[x]

                temp_positive_net=0
                temp_negative_net=0
                temp_neutral_net=0
                                    
                temp_positive_distance=0
                temp_negative_distance=0
                temp_neutral_distance=0
                
                temp_positive_likelehood=0
                temp_negative_likelehood=0
                temp_neutral_likelehood=0
                
                temp_z_positive=0
                temp_z_negative=0
                temp_z_neutral=0

                temp_z_positive_distance=0
                temp_z_negative_distance=0
                temp_z_neutral_distance=0
                
                for y in range(len(pol_sentence)):
                    pol_word=pol_sentence[y]
                    dist_word=distance[y]
                    neg_word=neg_sentence[y]
                    
                    if pol_word > 0:
                        temp_positive_net=temp_positive_net+1
                        temp_positive_distance=temp_positive_distance+1*dist_word
                        temp_positive_likelehood=temp_positive_likelehood+1*self.likelehood.get(neg_word,[0,0,0])[0]
                        temp_z_positive=temp_z_positive+self.z_dict_positive.get(neg_word,0)
                        temp_z_positive_distance=temp_z_positive_distance+self.z_dict_positive.get(neg_word,0)*dist_word


                    elif pol_word ==0:
                        temp_neutral_net=temp_neutral_net+1
                        temp_neutral_distance=temp_neutral_distance+1*dist_word
                        temp_neutral_likelehood=temp_neutral_likelehood+1*self.likelehood.get(neg_word,[0,0,0])[2]
                        temp_z_neutral=temp_z_neutral+self.z_dict_neutral.get(neg_word,0)
                        temp_z_neutral_distance=temp_z_neutral_distance+self.z_dict_neutral.get(neg_word,0)*dist_word

                    else:
                        temp_negative_net=temp_negative_net+1
                        temp_negative_distance=temp_negative_distance+1*dist_word
                        temp_negative_likelehood=temp_negative_likelehood+1*self.likelehood.get(neg_word,[0,0,0])[1]                                
                        temp_z_negative=temp_z_negative+self.z_dict_negative.get(neg_word,0)
                        temp_z_negative_distance=temp_z_negative_distance+self.z_dict_negative.get(neg_word,0)*dist_word

                
                likelehood_sum=(temp_positive_likelehood+temp_negative_likelehood+temp_neutral_likelehood)
                            
                temp_positive_likelehood=temp_positive_likelehood/likelehood_sum
                temp_negative_likelehood=temp_negative_likelehood/likelehood_sum
                temp_neutral_likelehood=temp_neutral_likelehood/likelehood_sum
                
                sentence_length=len(neg_sentence)
                
                net_positive.append(temp_positive_net)
                net_negative.append(temp_negative_net)
                net_neutral.append(temp_neutral_net)
                    
                net_positive_normalized.append(temp_positive_net/sentence_length)
                net_negative_normalized.append(temp_negative_net/sentence_length)
                net_neutral_normalized.append(temp_neutral_net/sentence_length)

                net_positive_distance.append(temp_positive_distance)
                net_negative_distance.append(temp_negative_distance)
                net_neutral_distance.append(temp_neutral_distance)

                net_positive_likelehood.append(temp_positive_likelehood)
                net_negative_likelehood.append(temp_negative_likelehood)
                net_neutral_likelehood.append(temp_neutral_likelehood)                

                z_positive.append(temp_z_positive)
                z_negative.append(temp_z_negative)
                z_neutral.append(temp_z_neutral)                
                
                z_positive_distance.append(temp_z_positive_distance)
                z_negative_distance.append(temp_z_negative_distance)
                z_neutral_distance.append(temp_z_neutral_distance)                

                temp_negative_net=0
                temp_negative_distance=0
                temp_negative_likelehood=0
                temp_z_negative=0

                temp_positive_net=0
                temp_positive_distance=0
                temp_positive_likelehood=0
                temp_z_positive=0
                
                temp_neutral_net=0
                temp_neutral_distance=0
                temp_neutral_likelehood=0                
                temp_z_neutral=0
                
                naive_prob_positive.append(self.naive.naive_features(neg_sentence)[0])
                naive_prob_negative.append(self.naive.naive_features(neg_sentence)[1])
                naive_prob_neutral.append(self.naive.naive_features(neg_sentence)[2])


            self.feature_matrix["net_positive"+str(self.partioned_boolean)]=net_positive
            self.feature_matrix["net_negative"+str(self.partioned_boolean)]=net_negative
            self.feature_matrix["net_neutral"+str(self.partioned_boolean)]=net_neutral
            self.feature_matrix["net_sentiment_avg"+str(self.partioned_boolean)]= (self.feature_matrix["net_positive"+str(self.partioned_boolean)]-1*self.feature_matrix["net_negative"+str(self.partioned_boolean)])/2
                        
            self.feature_matrix["net_positive_normalized"+str(self.partioned_boolean)]=net_positive_normalized
            self.feature_matrix["net_negative_normalized"+str(self.partioned_boolean)]=net_negative_normalized
            self.feature_matrix["net_neutral_normalized"+str(self.partioned_boolean)]=net_neutral_normalized
            self.feature_matrix["net_sentiment_normalized_avg"+str(self.partioned_boolean)]= (self.feature_matrix["net_positive_normalized"+str(self.partioned_boolean)]-1*self.feature_matrix["net_negative_normalized"+str(self.partioned_boolean)])/2
            
            self.feature_matrix["net_positive_likelehood"+str(self.partioned_boolean)]=net_positive_likelehood
            self.feature_matrix["net_negative_likelehood"+str(self.partioned_boolean)]=net_negative_likelehood
            self.feature_matrix["net_neutral_likelehood"+str(self.partioned_boolean)]=net_neutral_likelehood

            self.feature_matrix["z_positive"+str(self.partioned_boolean)]=(z_positive-np.mean(z_positive))/np.std(z_positive)
            self.feature_matrix["z_negative"+str(self.partioned_boolean)]=(z_negative-np.mean(z_negative))/np.std(z_negative)
            self.feature_matrix["z_neutral"+str(self.partioned_boolean)]=(z_neutral-np.mean(z_neutral))/np.std(z_neutral)
                        
            self.feature_matrix["naive_prob_positive"+str(self.partioned_boolean)]=naive_prob_positive
            self.feature_matrix["naive_prob_negative"+str(self.partioned_boolean)]=naive_prob_negative
            self.feature_matrix["naive_prob_neutral"+str(self.partioned_boolean)]=naive_prob_neutral

            if self.partioned_boolean==False:

                self.feature_matrix["net_positive_distance"+str(self.partioned_boolean)]=net_positive_distance
                self.feature_matrix["net_negative_distance"+str(self.partioned_boolean)]=net_negative_distance
                self.feature_matrix["net_neutral_distance"+str(self.partioned_boolean)]=net_neutral_distance

                self.feature_matrix["net_positive_distance_normalized"+str(self.partioned_boolean)]=net_positive_distance/self.feature_matrix["sentence_length"+str(self.partioned_boolean)]
                self.feature_matrix["net_negative_distance_normalized"+str(self.partioned_boolean)]=net_negative_distance/self.feature_matrix["sentence_length"+str(self.partioned_boolean)]
                self.feature_matrix["net_neutral_distance_normalized"+str(self.partioned_boolean)]=net_neutral_distance/self.feature_matrix["sentence_length"+str(self.partioned_boolean)]
                
                self.feature_matrix["z_positive_distance"+str(self.partioned_boolean)]=z_positive_distance
                self.feature_matrix["z_negative_distance"+str(self.partioned_boolean)]=z_negative_distance
                self.feature_matrix["z_neutral_distance"+str(self.partioned_boolean)]=z_neutral_distance


                self.feature_matrix["final_score_adv"+str(self.partioned_boolean)]=final_score_adv
                self.feature_matrix["final_score_adj"+str(self.partioned_boolean)]=final_score_adj
