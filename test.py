import pickle

#load the model and vectorizer
model = pickle.load(open('model.pkl','rb'))
word_vectorizer = pickle.load(open('vectorizer.pkl','rb'))

#test the model
print(model.predict(word_vectorizer.transform(["python AI machine learning deep learning"])))