#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('darkgrid') # darkgrid, white grid, dark, white and ticks
plt.rc('axes', titlesize=18)     # fontsize of the axes title
plt.rc('axes', labelsize=14)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=13)    # fontsize of the tick labels
plt.rc('ytick', labelsize=13)    # fontsize of the tick labels
plt.rc('legend', fontsize=13)    # legend fontsize
plt.rc('font', size=13)          # controls default text sizes


# # Preparing Data

# Goal: Our goal is to find which machine learning model is best suited to predict sentiment (output) given a movie review (input).
# 
# - Input(x) -> movie review
# - Ourput(y) -> sentiment  

# ## Reading dataset

# In[2]:


import pandas as pd

df_review = pd.read_csv('IMDB Dataset.csv')
df_review


# In[3]:


#taking a smaller sample of 10000 rows to make processing faster and get imbalance data
# 9000 positives
df_positive = df_review[df_review['sentiment']=='positive'][:9000]
# 1000 positives
df_negative = df_review[df_review['sentiment']=='negative'][:1000]

df_review_imb = pd.concat([df_positive, df_negative])
df_review_imb.value_counts(['sentiment'])


# ## Dealing with Imbalanced Classes

# In[4]:


colors = sns.color_palette('deep')

plt.figure(figsize=(8,4), tight_layout=True)
plt.bar(x=['Positive', 'Negative'],
        height=df_review_imb.value_counts(['sentiment']),
        color=colors[:2])
plt.title('Sentiment')
plt.savefig('sentiment.png')
plt.show()


# Different options to balanced data
# 
# 1. Python imbalanced-learn module
# 2. DataFrame.sample

# In[5]:


from imblearn.under_sampling import  RandomUnderSampler

rus = RandomUnderSampler(random_state=0)
df_review_bal, df_review_bal['sentiment']=rus.fit_resample(df_review_imb[['review']],
                                                           df_review_imb['sentiment'])
df_review_bal

# option 2
# length_negative = len(df_review[df_review['sentiment']=='negative'])
# df_review_positive = df_review[df_review['sentiment']=='positive'].sample(n=length_negative)
# df_review_non_positive = df_review[~(df_review['sentiment']=='positive')]

# df_review = pd.concat([
#     df_review_positive, df_review_non_positive
# ])
# df_review['sentiment'].value_counts()

# df_review.reset_index(drop=True, inplace=True)


# In[6]:


print(df_review_imb.value_counts('sentiment'))
print(df_review_bal.value_counts('sentiment'))


# ## Splitting data into train and test

# In[7]:


from sklearn.model_selection import train_test_split

train, test = train_test_split(df_review_bal, test_size=0.33, random_state=42)


# In[8]:


train_x, train_y = train['review'], train['sentiment']
test_x, test_y = test['review'], test['sentiment']


# In[9]:


train_y.value_counts()


# ## Count Vectorizer

# In[10]:


import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
text = ["I love writing code in Python. I love Python code",
        "I hate writing code in Java. I hate Java code"]

df = pd.DataFrame({'review': ['review1', 'review2'], 'text':text})
cv = CountVectorizer(stop_words='english')
cv_matrix = cv.fit_transform(df['text'])
df_dtm = pd.DataFrame(cv_matrix.toarray(), index=df['review'].values, columns=cv.get_feature_names())
df_dtm


# ## Tfidf

# ## Turning our text data into numerical vectors

# In[12]:


from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(stop_words='english')
train_x_vector = tfidf.fit_transform(train_x)
# also fit the test_x_vector
test_x_vector = tfidf.transform(test_x)

# from sklearn.feature_extraction.text import CountVectorizer

# cv = CountVectorizer(stop_words='english')
# train_x_vector = cv.fit_transform(train_x)
# test_x_vector = cv.transform(test_x)


# In[13]:


train_x_vector


# In[15]:


pd.DataFrame.sparse.from_spmatrix(train_x_vector,
                                  index=train_x.index,
                                  columns=tfidf.get_feature_names())


# ## Support Vector Machines (SVM)

# In[16]:


from sklearn.svm import SVC

svc = SVC(kernel='linear')
svc.fit(train_x_vector, train_y)
# svc.predict(train_x_vector[0])


# ### Testing

# In[17]:


print(svc.predict(tfidf.transform(['A good movie'])))
print(svc.predict(tfidf.transform(['An excellent movie'])))
print(svc.predict(tfidf.transform(['"I did not like this movie at all I gave this movie away"'])))


# ## Decision Tree

# In[18]:


from sklearn.tree import DecisionTreeClassifier

dec_tree = DecisionTreeClassifier()
dec_tree.fit(train_x_vector, train_y)


# ## Naive Bayes

# In[19]:


from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
gnb.fit(train_x_vector.toarray(), train_y)


# ## Logistic Regression

# In[20]:


from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression()
log_reg.fit(train_x_vector, train_y)


# # Model Evaluation

# ## Mean Accuracy

# Return the mean accuracy on the given test data and labels.

# In[21]:


print(svc.score(test_x_vector, test_y))
print(dec_tree.score(test_x_vector, test_y))
print(gnb.score(test_x_vector.toarray(), test_y))
print(log_reg.score(test_x_vector, test_y))

# svc.score('Test samples', 'True labels')


# ## F1 Score

# F1 Score is the weighted average of Precision and Recall. Accuracy is used when the True Positives and True negatives are more important while F1-score is used when the False Negatives and False Positives are crucial. Also, F1 takes into account how the data is distributed, so it's useful when you have data with imbalance classes.
# 
# F1 Score = 2*(Recall * Precision) / (Recall + Precision)

# In[22]:


from sklearn.metrics import f1_score

f1_score(test_y, svc.predict(test_x_vector),
         labels=['positive', 'negative'],
         average=None)

# f1_score(y_true, y_pred, average=None)


# ## Classification report

# Build a text report showing the main classification metrics.

# In[23]:


from sklearn.metrics import classification_report

print(classification_report(test_y, 
                            svc.predict(test_x_vector),
                            labels=['positive', 'negative']))
# classification_report(y_true, y_pred)


# ## Confusion Matrix

# A confusion matrix) is a table with two rows and two columns that reports the number of false positives, false negatives, true positives, and true negatives

# In[24]:


from sklearn.metrics import confusion_matrix

conf_mat = confusion_matrix(test_y, 
                            svc.predict(test_x_vector), 
                            labels=['positive', 'negative'])
conf_mat


# # Tuning the Model

# ## GridSearchCV

# In[25]:


from sklearn.model_selection import GridSearchCV

parameters = {'C': [1,4,8,16,32] ,'kernel':['linear', 'rbf']}
svc = SVC()
svc_grid = GridSearchCV(svc,parameters, cv=5,)
#              refit=True, verbose=0)
svc_grid.fit(train_x_vector, train_y)


# In[26]:


print(svc_grid.best_params_)
print(svc_grid.best_estimator_)


# In[ ]:


print(svc_grid.best_score_)

