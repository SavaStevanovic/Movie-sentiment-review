from tensorflow.keras.preprocessing.text import Tokenizer
import sys
import re

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

        for stopword in self.stopwords:
            series=series.str.replace(stopword, ' ')
        
        return(series)

    def tokenize(self):
        all_reviews = self.trainData.append(self.testData)
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(all_reviews)
        tokenizer.fit_on_sequences(all_reviews)

        self.trainData = tokenizer.texts_to_sequences(self.trainData)
        self.testData = tokenizer.texts_to_sequences(self.testData)

        self.trainData = tokenizer.sequences_to_matrix(self.trainData)
        self.testData = tokenizer.sequences_to_matrix(self.testData)
        # danse to sparse
