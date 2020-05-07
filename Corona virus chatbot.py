#!/usr/bin/env python
# coding: utf-8

# In[2]:


pip install newspaper3k


# In[3]:


from newspaper import Article
import random
import string 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import numpy as np
import warnings
warnings.filterwarnings('ignore')


# In[4]:


nltk.download('punkt', quiet=True) # Download the punkt package
nltk.download('wordnet', quiet=True)


# In[5]:


#Get the article URL
article = Article('https://www.medicalnewstoday.com/articles/256521')
article.download() #Download the article
article.parse() #Parse the article
article.nlp() #Apply Natural Language Processing (NLP)
corpus = article.text


# In[6]:


print(corpus)


# In[7]:


text = corpus
sent_tokens = nltk.sent_tokenize(text)


# In[8]:


remove_punct_dict = dict(  (ord(punct), None) for punct in string.punctuation)


# In[9]:


def LemNormalize(text):
    return nltk.word_tokenize(text.lower().translate(remove_punct_dict))


# In[10]:


# Keyword Matching
#Greeting input from the user
GREETING_INPUTS = ["hi", "hello",  "hola", "greetings",  "wassup","hey"] 
#Greeting responses back to the user
GREETING_RESPONSES = ["howdy","hi", "hey", "what's good",  "hello","hey there"]
#Function to return a random greeting response to a users greeting
def greeting(sentence):
   #If user's input is a greeting, return a randomly chosen greeting response
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)


# In[25]:


def response(user_response):
    robo_response='' #Create an empty response for the bot
    sent_tokens.append(user_response) #Append the users response to the list of sentence tokens
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english') 
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx=vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    score = flat[-2]
    if(score==0):
        robo_response=robo_response+"I apologize, I don't understand."
    else:
        robo_response = robo_response+sent_tokens[idx]+sent_tokens[idx+1]+sent_tokens[idx+2]+sent_tokens[idx+3]
    sent_tokens.remove(user_response) 
       
    return robo_response


# In[32]:


flag=True
print("Doctor: I will answer your queries about corona") 
engine = pyttsx3.init() 
while(flag==True):
    user_response = input()
    user_response=user_response.lower()
    if(user_response!='bye'):
        if(user_response=='thanks' or user_response=='thank you' ):
            flag=False
            print("Doctor: You're welcome !") 
        else:
            if(greeting(user_response)!=None):
                a=greeting(user_response)
                print("Doctor: "+a)
                engine.say(a) 
                engine.runAndWait() 
            else:
                print("Doctor: "+response(user_response))
                engine.say(response(user_response)) 
                engine.runAndWait() 
    else:
        flag=False
        print("Doctor: Chat with you later !")
        engine.say("Chat with you later !") 
        engine.runAndWait() 


# In[27]:


pip install gTTS


# In[28]:


# Import the required module for text  
# to speech conversion 
from gtts import gTTS 
  
# This module is imported so that we can  
# play the converted audio 
import os 
  
# The text that you want to convert to audio 
mytext = 'Welcome to geeksforgeeks!'
  
# Language in which you want to convert 
language = 'en'
  
# Passing the text and language to the engine,  
# here we have marked slow=False. Which tells  
# the module that the converted audio should  
# have a high speed 
myobj = gTTS(text=mytext, lang=language, slow=False) 
  
# Saving the converted audio in a mp3 file named 
# welcome  
myobj.save("welcome.mp3") 
  
# Playing the converted file 
os.system("mpg321 welcome.mp3") 


# In[29]:


pip install pyttsx3


# In[30]:


import pyttsx3 
  


# In[ ]:




