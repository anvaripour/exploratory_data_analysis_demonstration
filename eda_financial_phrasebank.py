#!/usr/bin/env python
# coding: utf-8

# In[41]:


# Import the necessary libraries:

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt2
import matplotlib.pyplot as plt3
from textblob import TextBlob
from datasets import load_dataset


# In[42]:


# load the dataset:

dataset = load_dataset('financial_phrasebank', 'sentences_allagree')
# dataset2 = load_dataset('financial_phrasebank', 'sentences_75agree')


# In[43]:


# Print the dataset info
print('features = ',dataset['train'].features)

# Print the first five rows of the dataset
print('first 5th = ',dataset['train'][:5])

# Get the sentiment label distribution
sentiment_distribution = dataset['train'].features['label'].str2int

# Print the sentiment label distribution
print('sentiment_distribution =', sentiment_distribution)


# In[44]:


# Check for missing data in the dataframe and remove it:

df = pd.DataFrame(dataset['train'])
df = df[~df['sentence'].isnull()]


# In[45]:


# Calculate sentiment polarity using textblob. The resulted values are in the range of [-1,1] 
# where 1 means positive sentiment and -1 means a negative sentiment.

df['polarity'] = df['sentence'].map(lambda text: TextBlob(text).sentiment.polarity)

# Print 5 random reviews with the sentiment polarity score over 0.4. We can check if label and polarity 
#are showing the same sentence tonality or not

cl = df.loc[df.polarity >0.4, ['sentence']].sample(5).values
for sentence in cl:
    print(sentence[0])
    index = df.index[df['sentence'] == sentence[0]][0]
    lab = df.loc[index, 'label']
    print("label=", lab, "df.polarity=", df.polarity[index])
    
sns.histplot(data=df, x='polarity', bins=50)
plt.xlabel('Polarity')
plt.ylabel('Count')
plt.show()


# In[46]:


# Compare the label and polarity distribution when polarity is categorized in three bins 

plt2.hist(df['polarity']+1, bins=3, alpha=0.5, label='polarity')
plt2.hist(df['label'], bins=3, alpha=0.5, label='label')
plt2.xlabel('Label and polarity distribution')
plt2.ylabel('Polarity vs label')
plt2.legend(loc='upper right')
plt2.show()


# In[27]:


# Compare the effect of the threshold on polarity distribution with label. By changing the threshold we can see
# how much the distribution can be similar to the label

new_polarity_0 = 0
new_polarity_1 = 0
new_polarity_2 = 0
threshold = 0.3
for cnt in df['polarity']:
    if cnt<-threshold:
        new_polarity_0 += 1
    elif -threshold <= cnt <= threshold:
        new_polarity_1 += 1
    else:
        new_polarity_2 += 1
        
new_label_0 = 0
new_label_1 = 0
new_label_2 = 0

for cnt in df['label']:
    if cnt == 0:
        new_label_0 += 1
    elif cnt == 1:
        new_label_1 += 1
    else:
        new_label_2 += 1

new_polarity = [new_polarity_0, new_polarity_1, new_polarity_2]
new_label = [new_label_0, new_label_1, new_label_2]
X = [0, 1, 2]
p1 = plt3.bar(X, new_polarity, 0.35, color='b',label='polarity')
X_plus_float = [x + 0.35 for x in X]
p2 = plt3.bar(X_plus_float, new_label, 0.35, color='g',label='label')
plt2.ylabel('Polarity counts on threshold %f ' %threshold)
plt2.legend(loc='upper right')
plt3.show()


# In[32]:


# The effect of a specific word on label

word_label = []
word_count = 0
specific_word = 'increase'
for index, row in df.iterrows():
    if specific_word in row['sentence']:
        word_count += 1
        word_label.append(row['label'])

x_labels = [0, 1, 2]
counts = [word_label.count(x) for x in x_labels]
plt.bar(x_labels, counts)

plt.xlabel("Label")
plt.ylabel("Specific word (%s) count in different labels" %specific_word)
plt.show()        
print('Number of sentences containing the word %s:' %specific_word, word_count)
print('Labels of the sentences containing the word:', word_label)


# In[33]:


# Get the sentence lengths and plot the distribution of sentence lengths

sentence_lengths = [len(sentence['sentence'].split()) for sentence in dataset['train']]
plt.hist(sentence_lengths, bins=50)
plt.xlabel('Sentence Length')
plt.ylabel('Frequency')
plt.show()


# In[36]:


# Create a dataframe with the sentence lengths and labels
labels = dataset['train']['label']
df = pd.DataFrame({'sentence_length': sentence_lengths, 'label': labels})


# In[37]:


# Get the sentence lengths for each label

lengths_0 = [len(sentence['sentence'].split()) for sentence in dataset['train'] if sentence['label'] == 0]
lengths_1 = [len(sentence['sentence'].split()) for sentence in dataset['train'] if sentence['label'] == 1]
lengths_2 = [len(sentence['sentence'].split()) for sentence in dataset['train'] if sentence['label'] == 2]

# Plot the histograms

plt.hist(lengths_0, bins=50, alpha=0.5, label='Label 0')
plt.hist(lengths_1, bins=50, alpha=0.5, label='Label 1')
plt.hist(lengths_2, bins=50, alpha=0.5, label='Label 2')

plt.xlabel('Sentence Length')
plt.ylabel('Frequency')
plt.legend(loc='upper right')
plt.show()


# In[38]:


# Plot the normalized histograms and compare the effect of the word count on the sentence tonality (labels)
plt.hist(lengths_0, bins=50, alpha=0.5, label='Label 0', density=True)
plt.hist(lengths_1, bins=50, alpha=0.5, label='Label 1', density=True)
plt.hist(lengths_2, bins=50, alpha=0.5, label='Label 2', density=True)

plt.xlabel('Sentence Length')
plt.ylabel('Frequency')
plt.legend(loc='upper right')
plt.show()


# In[ ]:




