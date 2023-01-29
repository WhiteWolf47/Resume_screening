import pickle
import shelve

#load the model and vectorizer
model = pickle.load(open('model.pkl','rb'))
word_vectorizer = pickle.load(open('vectorizer.pkl','rb'))

'''with shelve.open('model') as db:
    model = db['model']
    word_vectorizer = db['vectorizer']'''

#test the model
print(model.predict(word_vectorizer.transform(["python AI machine learning deep learning"])))