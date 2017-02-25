from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

class ClassifierData:
    X_train_tfidf=''
    target=''
    count_vect = CountVectorizer()
    tfidf_transformer = TfidfTransformer()

    def __init__(self,X_train_tfidf,target,count_vect,tfidf_transformer):
        self.X_train_tfidf=X_train_tfidf
        self.target=target
        self.count_vect=count_vect
        self.tfidf_transformer=tfidf_transformer
    def getX_train_tfidf(self):
        return self.X_train_tfidf
    def getTarget(self):
        return self.target
    def getCountVect(self):
        return self.count_vect
    def getIdfVector(self):
        return self.tfidf_transformer
