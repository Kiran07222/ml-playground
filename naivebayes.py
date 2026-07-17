from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
data = {

    "Message":[
        "Win a free lottery ticket",
        "Free money offer just for you",
        "Meeting at 10am tomorrow",
        "Project deadline extended",
        "Congratulations you won a prize"],
                
    "Spam": [1, 1, 0, 0, 1]
    }


df=pd.DataFrame(data)
vectorizer=CountVectorizer()
X= vectorizer.fit_transform(df["Message"])
y=df["Spam"]
model=MultinomialNB()
model.fit(X,y)
probabilites=model.predict_proba(X)
print(probabilites)
