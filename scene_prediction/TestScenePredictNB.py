import cPickle

from ScenePreditNB import ScenePredictNB

nb = ScenePredictNB()
#data loading
#loading training data
'''
with open('/home/semicolon/DRISHTI/DRISHTI/training_data/meeting_data_aws', 'r') as myfile:
    meeting_data = myfile.read().replace('\n', '')
with open('/home/semicolon/DRISHTI/DRISHTI/training_data/cafeteria_data_aws', 'r') as myfile:
    cafeteria_data = myfile.read().replace('\n', '')
with open('/home/semicolon/DRISHTI/DRISHTI/training_data/park_data_aws', 'r') as myfile:
    park_data = myfile.read().replace('\n', '')
with open('/home/semicolon/DRISHTI/DRISHTI/training_data/parking_lot_data_aws', 'r') as myfile:
    parking_lot_data = myfile.read().replace('\n', '')

data = [meeting_data, cafeteria_data, park_data, parking_lot_data]
target = ['meeting', 'cafeteria', 'park', 'parking_lot']
nb.train(data,target)
'''

print (nb.predict(list=['employee','office','people']))

