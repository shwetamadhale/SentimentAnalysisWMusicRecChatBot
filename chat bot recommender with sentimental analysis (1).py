#!/usr/bin/env python
# coding: utf-8

# In[3]:


get_ipython().system('pip install anvil-uplink')


# In[4]:


import anvil.server

anvil.server.connect("HU7U5DEGSJCT4NX4CNMXKXZO-4RBTIA7COKKOFH52")


# In[5]:


import numpy as np


# In[6]:


import tensorflow as tf


# In[7]:


from keras.models import Sequential


# In[8]:


from keras.layers import Dense, Activation, Dropout


# In[9]:


#optimizers
from tensorflow.keras.optimizers import SGD


# In[10]:


import random


# In[11]:


import nltk
from nltk.stem import WordNetLemmatizer


# In[12]:


lemmatizer = WordNetLemmatizer()


# In[13]:


import json
import pickle


# In[14]:


intents_file = open('C:/Users/madha/Downloads/Chatbot using Sentiment Analysis for song recommendation/intents.json').read()
intents = json.loads(intents_file)


# In[15]:


#tokenization and corpus
import nltk
nltk.download('punkt')
words = []
classes = []
documents = []
ignore_letters = ['!', '?', ',', '.']
for intent in intents['intents']:
    for pattern in intent['patterns']:
        #tokenizing each word
        word = nltk.word_tokenize(pattern)
        words.extend(word)
        #corpus
        documents.append((word, intent['tag']))
        #list classes
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
print(documents)


# In[16]:


#wordnet dictionary
import nltk
nltk.download('wordnet')
#training
training = []
#output
output_empty = [0] * len(classes)
#training and bagging sentence
for doc in documents:
    #initialization
    bag = []
    
    #list of tokenized words
    word_patterns = doc[0]
    
    #word root lemmatize
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    
    #bagging for matches
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)
        
    #0 op for each tag and 1 for currebt tag
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])
    
random.shuffle(training)
training = np.array(training)

#training and testing
train_x = list(training[:,0])
train_y = list(training[:,1])
print("Training data successful")


# In[17]:


#NN Model

model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

#Compiling, SGD and Nesterov accelerated gradient
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

#Training and saving the model 
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5', hist)

print("model is created")


# In[18]:


import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
from keras.models import load_model
model = load_model('chatbot_model.h5')
import json
import random
intents = json.loads(open('C:/Users/madha/Downloads/Chatbot using Sentiment Analysis for song recommendation/intents.json').read())
words = pickle.load(open('C:/Users/madha/Downloads/Chatbot using Sentiment Analysis for song recommendation/words.pkl','rb'))
classes = pickle.load(open('C:/Users/madha/Downloads/Chatbot using Sentiment Analysis for song recommendation/classes.pkl','rb'))
def clean_up_sentence(sentence):
    #tokenizing
    sentence_words = nltk.word_tokenize(sentence)
    #stemming to root
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words
#BOG array: 0 or 1 for words that exist in sentence
def bag_of_words(sentence, words, show_details=True):
    #tokenizing patterns
    sentence_words = clean_up_sentence(sentence)
    #bag of words - vocabulary matrix
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,word in enumerate(words):
            if word == s: 
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % word)
    return(np.array(bag))
def predict_class(sentence):
    #filter below  threshold predictions
    p = bag_of_words(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sorting strength probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list
def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result


# In[19]:


import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD
import random
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle

words=[]
classes = []
documents = []
ignore_letters = ['!', '?', ',', '.']
intents_file = open('C:/Users/madha/Downloads/Chatbot using Sentiment Analysis for song recommendation/intents.json').read()
intents = json.loads(intents_file)

for intent in intents['intents']:
    for pattern in intent['patterns']:
        #tokenizing
        word = nltk.word_tokenize(pattern)
        words.extend(word)
        #corpus
        documents.append((word, intent['tag']))
        #classes
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
print(documents)
#rooting and redundancy rmoval
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_letters]
words = sorted(list(set(words)))

#sorting
classes = sorted(list(set(classes)))

#documents = combination between patterns and intents
print (len(documents), "documents")

#classes = intents
print (len(classes), "classes", classes)

#words = all words, vocabulary
print (len(words), "unique lemmatized words", words)

pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))

#Training data
training = []
#output
output_empty = [0] * len(classes)

#training set and BOG Sentence
for doc in documents:
    # initialize our bag of words
    bag = []
    # list of tokenized words for the pattern
    pattern_words = doc[0]
    # lemmatize each word - create base word, in attempt to represent related words
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    # create our bag of words array with 1, if word match found in current pattern
    for word in words:
        bag.append(1) if word in pattern_words else bag.append(0)
        
    # output is a '0' for each tag and '1' for current tag (for each pattern)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    
    training.append([bag, output_row])
# shuffle our features and turn into np.array
random.shuffle(training)
training = np.array(training)
# create train and test lists. X - patterns, Y - intents
train_x = list(training[:,0])
train_y = list(training[:,1])
print("Training data created")

# Create model - 3 layers. First layer 128 neurons, second layer 64 neurons and 3rd output layer contains number of neurons
# equal to number of intents to predict output intent with softmax
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

#fitting and saving the model 
hist = model.fit(np.array(train_x), np.array(train_y), epochs=172, batch_size=5, verbose=1)
model.save('chatbot_model.h5', hist)

print("model created")


# In[20]:


import json
from ibm_watson import ToneAnalyzerV3
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator


# In[21]:


pip install --upgrade "ibm-watson>=5.3.0"


# In[22]:


import json
from ibm_watson import ToneAnalyzerV3
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator


# In[23]:


msg = list()
text = str()


# In[24]:


@anvil.server.callable
def responsed(msg1):
    msg.append(msg1)
    ints = predict_class(msg1)
    res = getResponse(ints, intents)
    return res


# In[ ]:


@anvil.server.callable
def song_emotion():
    authenticator = IAMAuthenticator("acgtQdBTLh1ul15KYi-6bfbP5EGtSC1OVrM6wgLooiLp")
    tone_analyzer = ToneAnalyzerV3(
        version='2017-09-21',
        authenticator=authenticator
    
    )

    tone_analyzer.set_service_url( "https://api.eu-gb.tone-analyzer.watson.cloud.ibm.com/instances/77cd4a81-e420-4212-85bc-3f4c4ae6dd4a")
    # text = ""
    # for i in msg:
    #     text = text+i
    len1 = len(msg)
    tone_analysis = tone_analyzer.tone(
        {'text': msg[len1-1]+" "+msg[len1-2]+" "+msg[len1-3]+" "+msg[len1-4]+" "+msg[len1-5]},
        content_type='application/json'
    ).get_result()
    dic1 = dict()
    emotion=tone_analysis["document_tone"]["tones"][0]["tone_name"]
    dic1['emotion'] = emotion
    import requests

    url=f"http://ws.audioscrobbler.com/2.0/?method=tag.gettoptracks&tag={emotion}&api_key=dc767f78f7b91f7b55d01c13fb684e31&format=json&limit=10"
    response = requests.get(url)
    payload = response.json()
    for i in range(10):
        r=payload['tracks']['track'][i]
        dic1[r['name']] = r['url']
    return dic1


# In[ ]:


import requests

url=f"http://ws.audioscrobbler.com/2.0/?method=tag.gettoptracks&tag=happy&api_key=dc767f78f7b91f7b55d01c13fb684e31&format=json&limit=5"
response = requests.get(url)
payload = response.json()
# for i in range(4):
r=payload['tracks']['track'][0]
# print(r['url'])
print(payload)


# In[ ]:


print("Chatbot : Hey there, Wassup ?")
# responded function takes text of user and returns chatbot output
for i in range(5):
    m = input("User : ")
    res = responsed(m)
    print("Chatbot : "+res)
ans = song_emotion()
print("Emotion : "+ans['emotion'])


# In[ ]:


ans = song_emotion()
print("Emotion : "+ans['emotion'])
ans.pop('emotion')
lst = list(ans.keys())
print("Song Recommendations : ")
for i in range(10):
    print("Song_name : "+lst[i])
    print("Song_URL : "+ans[lst[i]])

