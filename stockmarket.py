#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# هدف پیش بینی شاخص بانک با دیدن ویژگیهای نمادهای بانکی است. 
# روش پیشنهادی: RNN


# In[1]:


import pandas as pd
Data= pd.read_csv('Dataset.csv')
Data


# In[2]:


columns=Data.columns
columns


# In[3]:


Data.columns[33]


# In[4]:


columns=Data.columns[401:481]
columns


# In[5]:


DataFrame=Data.iloc[:, 401:481]
DataFrame['current']=Data['شاخص :: 57-بانكها']
# DataFrame['Date']=Data['تاریخ']



DataFrame
# DataFrame=Data


# In[6]:


import numpy as np
DataFrame.replace([np.inf, -np.inf], np.nan, inplace=True)
DataFrame.fillna(method="bfill", inplace=True)
DataFrame.fillna(method="ffill", inplace=True)
# if there are gaps in data, use previously known values
# DataFrame.dropna(inplace=True) 
print(DataFrame.head())  # how did we do??


# In[7]:


pd.isnull(DataFrame).sum() > 0


# In[8]:


DataFrame.isnull().sum()


# In[9]:


names=DataFrame.columns
names


# In[57]:




from sklearn import preprocessing 
from sklearn.preprocessing import StandardScaler
names = DataFrame.columns
scaler = preprocessing.StandardScaler()
scaled_df = scaler.fit_transform(DataFrame)
DataFrame = pd.DataFrame(scaled_df, columns=names)
# for col in DataFrame.columns: 
    
#         DataFrame[col] = preprocessing.scale(DataFrame[col].values)  # scale between 0 and 1.
# DataFrame['Date']=Data['تاریخ']    


# In[10]:


DataFrame


# In[11]:



SEQ_LEN = 60  # how long of a preceeding sequence to collect for RNN
FUTURE_PERIOD_PREDICT =5# how far into the future are we trying to predict?


# In[12]:


DataFrame['future'] = DataFrame['current'].shift(-FUTURE_PERIOD_PREDICT)


# In[13]:


# DataFrame['current']=DataFrame[f'{RATIO_TO_PREDICT}']


# In[14]:


DataFrame['current']


# In[15]:


DataFrame.head()


# In[16]:


DataFrame.tail(-90)


# In[17]:


def classify(current, future):
    if float(future) > float(current):
        return 1
    else:
        return 0


# In[18]:


DataFrame['target'] = list(map(classify, DataFrame['current'], DataFrame['future']))


# In[19]:


DataFrame['target']


# In[49]:





# In[20]:


import numpy as np
from collections import deque
import random
from sklearn import preprocessing  # pip install sklearn ... if you don't have it!


# In[21]:


times = sorted(DataFrame.index.values)  # get the times
last_5pct = sorted(DataFrame.index.values)[-int(0.05*len(times))]  # get the last 5% of the times

validation_DataFrame= DataFrame[(DataFrame.index >= last_5pct)]  # make the validation data where the index is in the last 5%
DataFrame = DataFrame[(DataFrame.index < last_5pct)]  # now the main_df is all the data up to the last 5%


# In[22]:


def preprocess_df(df):    
    df = df.drop("future", 1)  # don't need this anymore.

    for col in df.columns:  # go through all of the columns
        if col != "target":  # normalize all ... except for the target itself!
            df.dropna(inplace=True)  # remove the nas created by pct_change
            df[col] = preprocessing.scale(df[col].values)  # scale between 0 and 1.

    df.dropna(inplace=True)  # cleanup again... jic.

    sequential_data = []  # this is a list that will CONTAIN the sequences
    prev_days = deque(maxlen=SEQ_LEN)  # These will be our actual sequences. They are made with deque, which keeps the maximum length by popping out older values as new ones come in
    for i in df.values:  # iterate over the values
        prev_days.append([n for n in i[:-1]])  # store all but the target
        if len(prev_days) == SEQ_LEN:  # make sure we have 60 sequences!
            sequential_data.append([np.array(prev_days), i[-1]])  # append those bad boys!

    sequential_data

    random.shuffle(sequential_data)  # shuffle for good measure.
    buys = []  # list that will store our buy sequences and targets
    sells = []  # list that will store our sell sequences and targets
    for seq, target in sequential_data:  # iterate over the sequential data
        if target == 0:  # if it's a "not buy"
            sells.append([seq, target])  # append to sells list
        elif target == 1:  # otherwise if the target is a 1...
            buys.append([seq, target])  # it's a buy!


    random.shuffle(buys)  # shuffle the buys
    random.shuffle(sells)  # shuffle the sells!
    lower = min(len(buys), len(sells))  # what's the shorter length?

    buys = buys[:lower]  # make sure both lists are only up to the shortest length.
    sells = sells[:lower]  # make sure both lists are only up to the shortest length.

    sequential_data = buys+sells  # add them together
    random.shuffle(sequential_data)  # another shuffle, so the model doesn't get confused with all 1 class then the other.

    X = []
    y = []

    for seq, target in sequential_data:  # going over our new sequential data
        X.append(seq)  # X is the sequences
        y.append(target)  # y is the targets/labels (buys vs sell/notbuy)    

    return(np.array(X),y)


# In[23]:


DataFrame.tail()


# In[24]:



train_x, train_y = preprocess_df(DataFrame)
validation_x, validation_y = preprocess_df(validation_DataFrame)


# In[26]:


train_x.shape


# In[27]:


print(f"train data: {len(train_x)} validation: {len(validation_x)}")
print(f"Dont buys: {train_y.count(0)}, buys: {train_y.count(1)}")
print(f"VALIDATION Dont buys: {validation_y.count(0)}, buys: {validation_y.count(1)}")


# In[34]:


import time

EPOCHS = 5  # how many passes through our data
BATCH_SIZE = 64  # how many batches? Try smaller batch if you're getting OOM (out of memory) errors.


# In[29]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, LSTM, BatchNormalization


# In[35]:


model = Sequential()
model.add(LSTM(50, input_shape=(train_x.shape[1:]), return_sequences=True))
model.add(Dropout(0.2))
# model.add(BatchNormalization())  #normalizes activation outputs, same reason you want to normalize your input data.

model.add(LSTM(50, return_sequences=True))
model.add(Dropout(0.2)) #0.1
# model.add(BatchNormalization())

# =================
model.add(LSTM(50, return_sequences=True))
model.add(Dropout(0.1))

model.add(LSTM(50, return_sequences=True))
model.add(Dropout(0.2))


model.add(LSTM(50, return_sequences=True))
model.add(Dropout(0.1))


model.add(LSTM(50, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(50, return_sequences=True))
model.add(Dropout(0.1))


model.add(LSTM(50, return_sequences=True))
model.add(Dropout(0.2))


model.add(LSTM(50, return_sequences=True))
model.add(Dropout(0.1))


model.add(LSTM(50, return_sequences=True))
model.add(Dropout(0.2))
# ==================

model.add(LSTM(50)) #64 is hiperparameter
model.add(Dropout(0.2))
model.add(BatchNormalization())
# =====================

# =====================
model.add(Dense(32, activation='relu')) #32  is hiper parameter 
model.add(Dropout(0.2))


#one of these two lines must be chosen
model.add(Dense(2, activation='softmax'))
# model.add(Dense(1, activation='sigmoid'))


# In[39]:


opt = tf.keras.optimizers.Adam(lr=0.0001, decay=1e-6)

# Compile model
model.compile(
    loss='sparse_categorical_crossentropy',      #sparse would apply one hot encoding for target so target can be different values which did not pass one hot encoding inadvance
    optimizer=opt,
    metrics=['accuracy']                        #for balance dataset accuracy is good
)


# In[ ]:


# Train model
history = model.fit(
    train_x, np.array(train_y),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(validation_x, np.array(validation_y)))


# In[38]:


# Score model
score = model.evaluate(validation_x, np.array(validation_y), verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

