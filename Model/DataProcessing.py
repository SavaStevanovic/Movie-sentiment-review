from tensorflow.keras.preprocessing.text import Tokenizer
import sys
import re
import os
import csv
import operator
import numpy as np

class DataProcessing:
    stopwords = [ " a " ,  " about " ,  " above " ,  " after " ,  " again " ,  " against " ,  " all " ,  " am " ,  " an " ,  " and " ,  " any " ,  " are " ,  " as " ,  " at " ,  " be " ,  " because " ,  " been " ,  " before " ,  " being " ,  " below " ,  " between " ,  " both " ,  " but " ,  " by " ,  " could " ,  " did " ,  " do " ,  " does " ,  " doing " ,  " down " ,  " during " ,  " each " ,  " few " ,  " for " ,  " from " ,  " further " ,  " had " ,  " has " ,  " have " ,  " having " ,  " he " ,  " he'd " ,  " he'll " ,  " he's " ,  " her " ,  " here " ,  " here's " ,  " hers " ,  " herself " ,  " him " ,  " himself " ,  " his " ,  " how " ,  " how's " ,  " i " ,  " i'd " ,  " i'll " ,  " i'm " ,  " i've " ,  " if " ,  " in " ,  " into " ,  " is " ,  " it " ,  " it's " ,  " its " ,  " itself " ,  " let's " ,  " me " ,  " more " ,  " most " ,  " my " ,  " myself " ,  " nor " ,  " of " ,  " on " ,  " once " ,  " only " ,  " or " ,  " other " ,  " ought " ,
                  " our " ,  " ours " ,  " ourselves " ,  " out " ,  " over " ,  " own " ,  " same " ,  " she " ,  " she'd " ,  " she'll " ,  " she's " ,  " should " ,  " so " ,  " some " ,  " such " ,  " than " ,  " that " ,  " that's " ,  " the " ,  " their " ,  " theirs " ,  " them " ,  " themselves " ,  " then " ,  " there " ,  " there's " ,  " these " ,  " they " ,  " they'd " ,  " they'll " ,  " they're " ,  " they've " ,  " this " ,  " those " ,  " through " ,  " to " ,  " too " ,  " under " ,  " until " ,  " up " ,  " very " ,  " was " ,  " we " ,  " we'd " ,  " we'll " ,  " we're " ,  " we've " ,  " were " ,  " what " ,  " what's " ,  " when " ,  " when's " ,  " where " ,  " where's " ,  " which " ,  " while " ,  " who " ,  " who's " ,  " whom " ,  " why " ,  " why's " ,  " with " ,  " would " ,  " you " ,  " you'd " ,  " you'll " ,  " you're " ,  " you've " ,  " your " ,  " yours " ,  " yourself " ,  " yourselves " ]

    def __init__(self, trainData, testData, trainLable, testLable, remove_stopwords):
        self.trainData = trainData
        self.testData = testData
        self.trainLable = trainLable
        self.testLable = testLable
        self.remove_stopwords = remove_stopwords

    def getData(self):
        return self.trainData, self.testData, self.trainLable, self.testLable

    def cleanData(self):
        self.trainData = self.clean_text(self.trainData, self.remove_stopwords)
        self.testData = self.clean_text(self.testData, self.remove_stopwords)

    def clean_text(self, series, remove_stopwords=True):
        series=series.str.replace(r"<br />", ' ')
        print('Removing stopwords')
        if remove_stopwords:
            for stopword in self.stopwords:
                series=series.str.replace(stopword, ' ')
        print('Removed stopwords')
        
        return(series)

    def tokenize(self, writeDictionaryToCsv=False):
        print('Tokenizing')
        print('adding n-grams')
        self.trainData=self.addNgGrams(self.trainData)
        self.testData=self.addNgGrams(self.testData)

        all_reviews = self.trainData.append(self.testData)
        tokenizer = Tokenizer(num_words=9000)
        print('fitting')
        tokenizer.fit_on_texts(all_reviews)
        tokenizer.fit_on_sequences(all_reviews)

        print('texts_to_sequences')
        self.trainData = tokenizer.texts_to_sequences(self.trainData)
        self.testData = tokenizer.texts_to_sequences(self.testData)
        print('sequences_to_matrix')
        self.trainData = tokenizer.sequences_to_matrix(self.trainData)
        self.testData = tokenizer.sequences_to_matrix(self.testData)

        all_reviews=np.vstack((self.trainData, self.testData))
        self.trainData-=all_reviews.mean(axis=0)
        self.testData-=all_reviews.mean(axis=0)

        if(writeDictionaryToCsv):
            self.ExportFeatureSpace(tokenizer)

        tokenizer=None
        print('Finished tokenizing')

    def ExportFeatureSpace(self, tokenizer):
        print("Writing dictionary to csv")
        dirname = os.path.dirname(__file__)
        with open(dirname+'\\featurespace.csv', 'w', newline='' ) as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=['Word','Count'])
            writer.writeheader()
            for key, data in sorted(tokenizer.word_docs.items(), key=operator.itemgetter(1), reverse=True):
                writer.writerow({'Word':key.encode('utf-8').strip(),'Count':str(data)})
        print("Finished writing dictionary to csv")        

    def find_trigrams(self,input_list):
        return  [input_list[i]+"zzz" +input_list[i+1]+"zzz"+input_list[i+2] for i in range(len(input_list)-2)]  

    def find_bigrams(self,input_list):
        return [input_list[i]+"zzz" +input_list[i+1] for i in range(len(input_list)-1)]

    def addNgGrams(self, data):
        return data.apply(lambda x: self.addNGramsToSentance(x))

    def addNGramsToSentance(self, sentance):
        words = sentance.split(' ')
        bigrams=self.find_bigrams(words)
        trigrams=self.find_trigrams(words)
        extendedWords=words+bigrams+trigrams
        return ' '.join(extendedWords)
    