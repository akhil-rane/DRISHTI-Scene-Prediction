from ClassifierData import ClassifierData
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import cPickle
from aws import main as aws


'''
from sklearn import metrics
from sklearn.metrics import accuracy_score
from clarifai.rest import ClarifaiApp
from clarifai.rest import Image as ClImage
import json
import tensorflow.contrib.learn as skflow
import numpy
import os
import sys
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline
import time
'''

class ScenePredictNB:

    def __init__(self):
        ''' Constructor for this class.
        '''
        # Create some member animals

    def train(self,data,target):

        # training the model using tf-idf and naive bayes classifier
        count_vect = CountVectorizer()
        X_train_counts = count_vect.fit_transform(data)
        tfidf_transformer = TfidfTransformer()
        X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
        clf = MultinomialNB().fit(X_train_tfidf, target)

        #store model
        with open('scene_predict_nb.pickle', 'wb') as f:
            cPickle.dump(clf, f)
        classifierData=ClassifierData(X_train_tfidf,target,count_vect,tfidf_transformer)
        with open('classifier_data.pickle', 'wb') as cd:
            cPickle.dump(classifierData, cd)
        print ('Naive Bayes Model trained..')


    def predict(self,test_tags):

        with open('classifier_data.pickle', 'rb') as cd:
            classifierData = cPickle.load(cd)
        with open('scene_predict_nb.pickle', 'rb') as f:
            clf = cPickle.load(f)

        test_data = [test_tags]

        count_vect = classifierData.getCountVect()

        tfidf_transformer = classifierData.getIdfVector()

        X_new_counts = count_vect.transform(test_data)
        X_new_tfidf = tfidf_transformer.transform(X_new_counts)

        #clf = MultinomialNB().fit(classifierData.getX_train_tfidf(), classifierData.getTarget())
        predicted = clf.predict(X_new_tfidf)
        return predicted[0]

    def predict_image(self,image_file):

        output = aws(image_file)
        tags=''
        # print output
        for i in output:
            if i['Confidence'] > 0.70:
                tags = tags + i['Name'] + " "

        test_data = [tags]

        with open('classifier_data.pickle', 'rb') as cd:
            classifierData = cPickle.load(cd)
        with open('scene_predict_nb.pickle', 'rb') as f:
            clf = cPickle.load(f)

        count_vect = classifierData.getCountVect()

        tfidf_transformer = classifierData.getIdfVector()

        X_new_counts = count_vect.transform(test_data)
        X_new_tfidf = tfidf_transformer.transform(X_new_counts)

        #clf = MultinomialNB().fit(classifierData.getX_train_tfidf(), classifierData.getTarget())
        predicted = clf.predict(X_new_tfidf)
        return predicted[0]










'''
    def test(self):

        # loading images taken by pranav in office
        test_data = []
        test_target = []
        print 'asd'

        app = ClarifaiApp()

        for filename in os.listdir(self.test_images_folder_path+'/meeting'):
            image_file = [self.test_images_folder_path+'/meeting/'+filename]
            json_data= json.dumps(app.tag_files(image_file,model='aaa03c23b3724a16a56b629203edc62c'))
            json_obj=json.loads(json_data)
            json_tags=json_obj['outputs'][0]['data']['concepts']
            tags=''
            for tag in json_tags:
                if tag['value'] > 0.90 :
                    tags=tags+tag['name']+' '
            test_data.append(tags)
            test_target.append('meeting')

        for filename in os.listdir(self.test_images_folder_path+'/park'):
                image_file = [self.test_images_folder_path+'/park/'+filename]
                json_data= json.dumps(app.tag_files(image_file,model='aaa03c23b3724a16a56b629203edc62c'))
                json_obj=json.loads(json_data)
                json_tags=json_obj['outputs'][0]['data']['concepts']
                tags=''
                for tag in json_tags:
                    if tag['value'] > 0.90 :
                        tags=tags+tag['name']+' '
                test_data.append(tags)
                test_target.append('park')

        for filename in os.listdir(self.test_images_folder_path+'/parking_lot'):
                image_file = [self.test_images_folder_path+'/parking_lot/'+filename]
                json_data= json.dumps(app.tag_files(image_file,model='aaa03c23b3724a16a56b629203edc62c'))
                json_obj=json.loads(json_data)
                json_tags=json_obj['outputs'][0]['data']['concepts']
                tags=''
                for tag in json_tags:
                    if tag['value'] > 0.90 :
                        tags=tags+tag['name']+' '
                test_data.append(tags)
                test_target.append('parking_lot')

        #Performance evaluation with test data
        with open('scene_predict_nb.pickle', 'rb') as f:
            clf = cPickle.load(f)
        count_vect = CountVectorizer()
        tfidf_transformer = TfidfTransformer()
        X_new_counts = count_vect.transform(test_data)
        X_new_tfidf = tfidf_transformer.transform(X_new_counts)
        predicted = clf.predict(X_new_tfidf)
        print 'True Labels of test data:'
        print test_target
        print 'Predicted Labels of test data:'
        print predicted
        print 'Accuracy:'
        print (accuracy_score(test_target, predicted))

    def predict(self,file_path):

        # loading training data
        with open('/home/semicolon/DRISHTI/DRISHTI/training_data/meeting_data_bck', 'r') as myfile:
            meeting_data = myfile.read().replace('\n', '')
        with open('/home/semicolon/DRISHTI/DRISHTI/training_data/cafeteria_data_bck', 'r') as myfile:
            cafeteria_data = myfile.read().replace('\n', '')
        with open('/home/semicolon/DRISHTI/DRISHTI/training_data/park_data_bck', 'r') as myfile:
            park_data = myfile.read().replace('\n', '')
        with open('/home/semicolon/DRISHTI/DRISHTI/training_data/parking_lot_data_bck', 'r') as myfile:
            parking_lot_data = myfile.read().replace('\n', '')

        data = [meeting_data, cafeteria_data, park_data, parking_lot_data]
        target = ['meeting', 'cafeteria', 'park', 'parking_lot']


        #training the model using tf-idf and naive bayes classifier
        count_vect = CountVectorizer()
        X_train_counts = count_vect.fit_transform(data)
        tfidf_transformer = TfidfTransformer()
        X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
        feature_columns = skflow.infer_real_valued_columns_from_input(X_train_tfidf)
        clf = MultinomialNB().fit(X_train_tfidf, target)


        #Scene prediction for new input image

        image_file = [file_path]
        app = ClarifaiApp()
        json_data= json.dumps(app.tag_files(image_file,model='aaa03c23b3724a16a56b629203edc62c'))
        json_obj=json.loads(json_data)
        json_tags=json_obj['outputs'][0]['data']['concepts']
        tags=''
        for tag in json_tags:
            tags=tags+tag['name']+' '
        test_data=[tags]
        X_new_counts = count_vect.transform(test_data)
        X_new_tfidf = tfidf_transformer.transform(X_new_counts)
        predicted = clf.predict(X_new_tfidf)
        return predicted[0]

    def predict_image(self,image_file):
        # loading training data
        with open(self.meeting_data_path, 'r') as myfile:
            meeting_data = myfile.read().replace('\n', '')
        with open(self.cafeteria_data_path, 'r') as myfile:
            cafeteria_data = myfile.read().replace('\n', '')
        with open(self.park_data_path, 'r') as myfile:
            park_data = myfile.read().replace('\n', '')
        with open(self.parking_lot_data_path, 'r') as myfile:
            parking_lot_data = myfile.read().replace('\n', '')

        data = [meeting_data, cafeteria_data, park_data, parking_lot_data]
        target = ['meeting', 'cafeteria', 'park', 'parking_lot']

        # training the model using tf-idf and naive bayes classifier
        count_vect = CountVectorizer()
        X_train_counts = count_vect.fit_transform(data)
        tfidf_transformer = TfidfTransformer()
        X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
        clf = MultinomialNB().fit(X_train_tfidf, target)

        # Scene prediction for new input image

        image_file = [file_path]
        app = ClarifaiApp()
        json_data = json.dumps(app.tag_files(image_file, model='aaa03c23b3724a16a56b629203edc62c'))
        json_obj = json.loads(json_data)
        json_tags = json_obj['outputs'][0]['data']['concepts']
        tags = ''
        for tag in json_tags:
            tags = tags + tag['name'] + ' '
        test_data = [tags]
        X_new_counts = count_vect.transform(test_data)
        X_new_tfidf = tfidf_transformer.transform(X_new_counts)
        predicted = clf.predict(X_new_tfidf)
        return predicted[0]
'''