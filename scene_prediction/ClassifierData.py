from sklearn.feature_extraction.text import CountVectorizer

class ClassifierData:
    X_train_tfidf=''
    target=''
    count_vect = CountVectorizer()
    def __init__(self,X_train_tfidf,target,count_vect):
        self.X_train_tfidf=X_train_tfidf
        self.target=target
        self.count_vect=count_vect
    def getX_train_tfidf(self):
        return self.X_train_tfidf
    def getTarget(self):
        return self.target
    def getCountVect(self):
        return self.count_vect