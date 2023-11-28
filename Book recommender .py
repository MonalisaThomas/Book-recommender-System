#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import plotly.express as px
import plotly.graph_objects as go


#loading the data
df = pd.read_csv("books.csv", on_bad_lines='skip')
print(df.head())


# In[3]:


df=df[["title","average_rating","publication_date"]]
print(df.head(5))


# In[4]:


df = df.sort_values(by="average_rating", ascending=False)
top_5 = df.head()

labels = top_5["title"]
values = top_5["average_rating"]
colors = ['gold','lightgreen']


fig = go.Figure(data=[go.Pie(labels=labels, values=values)])
fig.update_layout(title_text="Top 5 Rated Books")
fig.update_traces(hoverinfo='label+percent', textinfo='percent', textfont_size=30,
                  marker=dict(colors=colors, line=dict(color='black', width=3)))
fig.show()


# In[5]:


print(df.isnull().sum())


# In[6]:


# Load the dataset
df = pd.read_csv('books.csv',on_bad_lines='skip')

# Combine relevant text features into a single string for each book
df['features'] = df['title'] + ' ' + df['authors'] + ' ' + df['language_code'] + ' ' + df['publisher']

# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer(stop_words='english')

# Transform the features into TF-IDF vectors
tfidf_matrix = vectorizer.fit_transform(df['features'])

# Calculate similarity between books using linear kernel (cosine similarity)
cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)

# Function to get book recommendations based on book title
def get_recommendations(book_title, cosine_similarities=cosine_similarities):
    # Check if the book title exists in the DataFrame
    if book_title not in df['title'].values:
        return "Book not found in the dataset."

    book_index = df[df['title'] == book_title].index[0]
    similarity_scores = list(enumerate(cosine_similarities[book_index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    
    # Exclude the book itself
    top_recommendations = similarity_scores[1:4]
    
    # Get the indices of recommended books
    recommended_indices = [idx for idx, _ in top_recommendations]
    
    # Return the titles of recommended books
    return df['title'].iloc[recommended_indices]

# Example: Get recommendations for a specific book
user_input = input("Enter a book title: ")
recommendations = get_recommendations(user_input)
print(f"Recommendations for '{user_input}':")
print(recommendations)


# In[ ]:





# In[ ]:




