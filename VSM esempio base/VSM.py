#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 18:32:26 2019

@author: marcocianciotta
"""
import pprint
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


"""
The first step in modeling the document into a vector space is to create a 
dictionary of terms present in documents. To do that, you can simple select all
terms from the document and convert it to a dimension in the vector space, but
we know that there are some kind of words (stop words) that are present 
in almost all documents, and what we’re doing is extracting important features
from documents, features do identify them among other similar documents, 
so using terms like “the, is, at, on”, etc.. isn’t going to help us, 
so in the information extraction, we’ll just ignore them.
"""

train_set = ("Il cielo è blu", "Il sole è luminoso")
test_set = ("Il sole nel cielo è luminoso.","Possiamo vedere il tramonto del sole, il sole è luminoso.")

"""
Now, what we have to do is to create a index vocabulary (dictionary) of the 
words of the train document set.
In scikit.learn, what we have presented as the term-frequency, 
is called CountVectorizer, so we need to import it and create a news instance.


The CountVectorizer already uses as default “analyzer” called WordNGramAnalyzer, 
which is responsible to convert the text to lowercase, accents removal, 
token extraction, filter stop words, etc… 
"""

count_vectorizer = CountVectorizer()
#print(count_vectorizer)

count_vectorizer.fit_transform(train_set)
print("Vocabolario\n", count_vectorizer.get_feature_names())
print ("\n------------------------------------------------")
#print ("Vocabulary:", count_vectorizer.vocabulary)


"""
since we have a collection of documents, now represented by vectors, 
we can represent them as a matrix.
As you may have noted, these matrices representing the term frequencies 
tend to be very sparse (with majority of terms zeroed).

"""
print("\ntest_set:\n", test_set)
print ("\n------------------------------------------------")

freq_term_matrix = count_vectorizer.transform(test_set)
print("\nMatrice della frequenza dei termini\n")
print (freq_term_matrix.todense())
print ("\n------------------------------------------------")

print ("\n")


from sklearn.feature_extraction.text import TfidfTransformer

"""
the most common norm used to measure the length of a vector, typically called “magnitude”;
a norm is a function that assigns a strictly positive length or size to all vectors in a vector space.
The process of using the L2-norm (we’ll use the right terms now) to normalize 
our vector in order to get its unit vector.

"""
tfidf = TfidfTransformer(norm="l2")
tfidf.fit(freq_term_matrix)

"""
What tf-idf gives is how important is a word to a document in a collection, 
and that’s why tf-idf incorporates local and global parameters, 
because it takes in consideration not only the isolated term but also 
the term within the document collection.
What tf-idf then does to solve that problem, is to scale down the frequent 
terms while scaling up the rare terms; a term that occurs 10 times more than 
another isn’t 10 times more important than it, that’s why tf-idf uses the l
ogarithmic scale to do that.

"""


print ("\n")


tf_idf_matrix = tfidf.transform(freq_term_matrix)
print ("Tf idf matrix\n")
print (tf_idf_matrix.todense())
print ("\n------------------------------------------------")


from sklearn.metrics.pairwise import cosine_similarity
print ("\nCosine Similarity between the first document and other \n (il primo valore è 1 poichè è la cosine similarity tra il 1 documento e se stesso)")
print("\n")
print(cosine_similarity(tf_idf_matrix[0:1], tf_idf_matrix))

print ("\n------------------------------------------------")
    

