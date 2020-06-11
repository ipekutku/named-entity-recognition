#importing necessary libraries
import pandas as pd
import collections
import numpy as np
import time

start = time.time()

class HiddenMarkovModel(object):
    
    #class init function, transitions, emissions, list of sentences all assigned here.
    def __init__(self, train_folderpath, test_folderpath):
        self.tag_counts = {}
        self.word_tag_counts = {}
        self.bigram_tag_counts = {}
        self.__trainPath = train_folderpath
        self.__testPath = test_folderpath
        self.train_sentences = self.__dataset(self.__trainPath)
        self.observations, self.gold_sequences = self.__dataset(self.__testPath, test=True)
        self.initial_probs, self.transition_probs, self.emission_probs = self.__HMM(self.train_sentences)
    
    
    #dataset function, takes folderpath and returns list of sentences with tags 
    def __dataset(self,folderpath, test=None):
        list_of_sentences = []
        words = []
        train = True
        if(test == True): # if test, the list of tags is seperated from words as ground truths 
            train = False
            list_of_ground_truths = []
            tags = []
        with open(folderpath) as f:
            for line in f:
                if(line != "\n"):
                    if(line == "-DOCSTART- -X- -X- O\n"): #ignore that line
                        continue
                    line_list = line.split()
                    word = line_list[0]
                    tag = line_list[-1]
                    if(train):
                        string = "\\".join([word,tag]) # for train, words and tags seperated by '\\'
                        words.append(string)
                    else:
                        words.append(word)
                        tags.append(tag)
                else:
                    if len(words) == 0:
                        continue
                    else:
                        if not (train):
                            tag_seq = " ".join(tags)
                            list_of_ground_truths.append(tag_seq)
                            tags = []
                        sentence = " ".join(words)
                        list_of_sentences.append(sentence)
                        words = []
        if(train):
            return list_of_sentences #for train.txt
        else:
            return list_of_sentences, list_of_ground_truths #for test.txt
    
    #creating Hidden Markov Model, it returns initial, transition and emission probs
    def __HMM(self, list_of_sentences):
        tag_counts = {} # dictionary for unigram tag counts.
        word_tag_counts = {} # dictionary for calculating emission probs
        bigram_tag_counts = {}  # dictionary for bigram tag counts, used to calculate transition probs
        for sentence in list_of_sentences:
            words = sentence.split()
            words.insert(0, '<s>\\<s>') # <s> is added to calculate initial probabilities
            words.append('</s>\\</s>') # </s> added as a end of sentence and end of tags token
            for i in range(len(words)):
                word_tag = words[i].split("\\")
                word = word_tag[0]
                tag = word_tag[1]
                tag_counts = self.__gen_ngram_dict(tag, tag_counts) #adding tag to dict
                word_tag_counts = self.__gen_ngram_dict((word,tag), word_tag_counts) #adding word and its tag to dict
                if(i<len(words)-1):
                    bigram_tag_counts = self.__gen_ngram_dict(tag, bigram_tag_counts, bi_word = words[i+1]) #adding tags to dict
        
        self.tag_counts = tag_counts # set class variable tag_counts
        self.word_tag_counts = word_tag_counts #set class variable word_tag_countss
        self.bigram_tag_counts = bigram_tag_counts # set class variable bigram_tag_counts
        weights = self.deleted_interpolation() #calculation of weigths for transitions using deleted interpolation
        transition_probs, initial_probs = self.__calc_transition_probs(weights, tag_counts, bigram_tag_counts) #calculation of transition end initial probs 
        emission_probs = self.__calc_emission_probs(tag_counts, word_tag_counts) # calculation of emission probs
        
        return initial_probs, transition_probs, emission_probs
    
    #viterbi algorithm, takes o_t(observation), inital, transition, emission probs, returns best tag path
    def viterbi(self, o_t, initial_probs, transitions, emissions):
        states = list(emissions.keys()) # state list
        states.remove('<s>') #<s> is for initial probs
        unique = len(self.word_tag_counts) #unique count for smoothed emission probabilities
        Viterbi = np.zeros((len(states), len(o_t))) #viterbi matrix
        backpointers = np.zeros((len(states), len(o_t)-1)) #backpointer matrix
    
        #initialization part
        idx = 0 #state index
        first_word = o_t[0] #first observation
        for state in states:
            initial_prob = np.log2(initial_probs[idx+1])
            try:
                emission = np.log2(emissions[state][first_word])
            except KeyError:
                emission = np.log2(1/(self.tag_counts[state]+unique))
                
            Viterbi[idx, 0] = initial_prob + emission
            idx += 1
            
        #from step 1 to T, viterbi and backpointer matrices are filled with appropriate values
        for step in range(1, len(o_t)):
            idx=0 # state index
            prev_col = Viterbi[:, step-1]
            word = o_t[step] #single observation
            for state in states:
                transition = np.log2(transitions[:, idx])
                try:
                    emission = np.log2(emissions[state][word])
                except KeyError:
                    emission = np.log2(1/(self.tag_counts[state]+unique))
                
                prob = prev_col + transition + emission # prob array for the current state
            
                backpointers[idx , step-1] = np.argmax(prob) #current state's most probable prev state index
                Viterbi[idx, step] = np.max(prob) #current state's max prob value  
                idx+=1
    
        #creating path list with last_state as its first element
        path = [None]*len(o_t)
        last_state = int(np.argmax(Viterbi[:, -1])) #last_state index
        path[0] = last_state
        
        #backtracking part
        back_index = 1
        for i in range(len(o_t)-2, -1, -1):
            path[back_index] = int(backpointers[last_state, i]) #adding tags to path list from backpointer matrix
            last_state = int(backpointers[last_state, i])
            back_index += 1
            
        path.reverse() # reverse path list 
    
        real_path = [] # list for tags as string, in path list, tags are indices
        
        for index in path:
            tag = states[index]
            real_path.append(tag)

        return real_path    
        
    #function to calculate accuracy
    def accuracy(self, ground_truths, prediction):
        total_word = 0
        true_preds = 0
        for i in range(len(ground_truths)):
            gold = ground_truths[i].split()
            pred = prediction[i][:-1]
            total_word += len(gold)
            for j in range(len(pred)):
                if gold[j] == pred[j]:
                    true_preds +=1  
        return (true_preds / total_word)
    
    
    ##### HELPER FUNCTIONS####
    
    #function for testing the model, returns the prediction list for all observations
    def test_model(self, observations, initial_probs, transitions, emissions):
        predictions = []
        for observation in observations: # for each test sentence
            #observation = observation.lower()
            o_list = observation.split()
            o_list.append("</s>")
            result = self.viterbi(o_list , initial_probs, transitions, emissions) # each sentence goes to viterbi
            #predictions.append(result[:-1])# prediction list
            predictions.append(result)
        return predictions
    
    #deleted interpolation algorithm
    def deleted_interpolation(self):
        lambda_1 = 0
        lambda_2 = 0
        for pair in self.bigram_tag_counts.keys():
            count = self.bigram_tag_counts[pair]
            value = self.tag_counts[pair[0]]
            c_1 = (count - 1) / (value - 1) #for bigram
            c_2 = (value - 1) / (sum(self.tag_counts.values()) - 1) #for unigram
            clist = [c_1, c_2]
            
            max_idx = np.argmax(clist)
            if max_idx == 0:
                lambda_2 += count
            if max_idx == 1:
                lambda_1 += count
    
        w = [lambda_1, lambda_2]
        w = [element / sum(w) for element in w] #normalization of weights
    
        return w
    
    #function for calculating initial and transition probs
    def __calc_transition_probs(self, weights, tag_counts, bigram_tag_counts):
        states = list(tag_counts.keys())
        df = pd.DataFrame(columns=states, index=states)
        for index in df.index:
            for column in df.columns:
                tag_set = (index,column)
                try:
                    prob = (weights[1] * (bigram_tag_counts[tag_set] / tag_counts[index])) + (weights[0]*(tag_counts[column] / sum(tag_counts.values())))
                except:
                    prob = weights[0]*(tag_counts[column] / sum(tag_counts.values()))
                
                df.loc[index,column] = prob
        initial_probs = df.loc['<s>', :] # initial probs
        df = df.drop(columns = '<s>', index = '<s>') # transition probs
        return df, initial_probs
    
    #function for calculating emission probs
    def __calc_emission_probs(self, tag_counts, word_tag_counts):
        emissions = collections.defaultdict(dict) # emission probs as a nested dictionary
        for key in word_tag_counts:
            word = key[0]
            tag = key[1]
            prob = (word_tag_counts[key] + 1) / (tag_counts[tag] + len(word_tag_counts)) #laplace smoothing
            emissions[tag][word] = prob
        return emissions
    
    #function to create n_gram tag dicts
    def __gen_ngram_dict(self, tag, tag_dict, bi_word=None):
        tag_set = tag
        if(bi_word != None):
            tag_2 = bi_word.split("\\")[1]
            tag_set = (tag, tag_2)
        try:
            tag_dict[tag_set] += 1
        except:
            tag_dict[tag_set] = 1
        return tag_dict
    
model = HiddenMarkovModel("train.txt", "test.txt") #creating the model

#necessary objects to evaluate the model
initial_probs = model.initial_probs
transitions = model.transition_probs
emissions = model.emission_probs
observations = model.observations
gold_seq = model.gold_sequences

predictions = model.test_model(observations, initial_probs.values.astype('float64'), transitions.values.astype('float64'), emissions)

accuracy = model.accuracy(gold_seq, predictions) #calculating accuracy

print("Accuracy = " + str(accuracy) + '\n')

end = time.time()
print("Time the program takes = " + str(end-start) + '\n')