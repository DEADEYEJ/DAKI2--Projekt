from sklearn.feature_extraction.text import TfidfVectorizer


d0 = "This is a sample A."
d1 = "This is another example, another example, example."


string = [d0, d1]


tfidf = TfidfVectorizer()
result = tfidf.fit_transform(string)


print('\nidf values:')
for ele1, ele2 in zip(tfidf.get_feature_names_out(), tfidf.idf_):
    print(ele1, ':', round(ele2, 3))

print('\nWord indexes:')
print(tfidf.vocabulary_)
print('\ntf-idf value:')
print(result)