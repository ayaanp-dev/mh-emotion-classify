import pandas as pd

tweet_emotions = pd.read_csv("./data/tweet_emotions.csv")
emotion_angry = pd.read_csv("./data/Emotion(angry).csv")["content"].to_list()
emotion_happy = pd.read_csv("./data/Emotion(happy).csv")["content"].to_list()
emotion_sad = pd.read_csv("./data/Emotion(sad).csv")["content"].to_list()
text = pd.read_csv("./data/text.csv")
text_emotion = pd.read_csv("./data/Text_Emotion.csv")

# print(tweet_emotions["sentiment"].unique())
# en1 = tweet_emotions.loc[(tweet_emotions["sentiment"] == "empty") | (tweet_emotions["sentiment"] == "sadness") | (tweet_emotions["sentiment"] == "worry") | (tweet_emotions["sentiment"] == "anger") | (tweet_emotions["sentiment"] == "hate")]["content"].to_list()
# en2 = emotion_angry + emotion_sad
# en3 = text.loc[(text["label"] == 0) | (text["label"] == 3) | (text["label"] == 4)]["text"].to_list()
# en4 = text_emotion[(text_emotion["emotion"] == "☹️")]["text"].to_list()

# negative = pd.DataFrame({"content": en1 + en2 + en3 + en4, "emotion": 0})

# ep1 = tweet_emotions.loc[(tweet_emotions["sentiment"] == "enthusiasm") | (tweet_emotions["sentiment"] == "love") | (tweet_emotions["sentiment"] == "fun") | (tweet_emotions["sentiment"] == "happiness") | (tweet_emotions["sentiment"] == "relief") | (tweet_emotions["sentiment"] == "fun")]["content"].to_list()
# ep2 = emotion_happy
# ep3 = text.loc[(text["label"] == 1) | (text["label"] == 2)]["text"].to_list()
# ep4 = text_emotion[(text_emotion["emotion"] == "🙂")]["text"].to_list()

# positive =  pd.DataFrame({"content": ep1 + ep2 + ep3 + ep4, "emotion": 1})

# negative.to_csv("../data/tri/negative.csv", index=False)
# positive.to_csv("../data/tri/positive.csv", index=False)

# neutral = tweet_emotions.loc[(tweet_emotions["sentiment"] == "neutral") | (tweet_emotions["sentiment"] == "boredom")]["content"].to_list()
# neutral = pd.DataFrame({"content": neutral, "emotion": 2})
# neutral.to_csv("../data/tri/neutral.csv", index=False)

# separate data into anger, sadness, happiness, neutral, and love
a1 = tweet_emotions.loc[(tweet_emotions["sentiment"] == "anger") | (tweet_emotions["sentiment"] == "hate")]["content"].to_list()
a2 = emotion_angry
a3 = text.loc[(text["label"] == 3)]["text"].to_list()
angry = pd.DataFrame({"content": a1 + a2 + a3, "emotion": 0})

s1 = text_emotion[(text_emotion["emotion"] == "☹️")]["text"].to_list()
s2 = emotion_sad
s3 = tweet_emotions.loc[(tweet_emotions["sentiment"] == "sadness")]["content"].to_list()
s4 = text.loc[(text["label"] == 0)]["text"].to_list()
sadness = pd.DataFrame({"content": s1 + s2 + s3 + s4, "emotion": 1})

h1 = emotion_happy
h2 = tweet_emotions.loc[(tweet_emotions["sentiment"] == "enthusiasm") | (tweet_emotions["sentiment"] == "fun")]["content"].to_list()
h3 = text.loc[(text["label"] == 1)]["text"].to_list()
h4 = text_emotion[(text_emotion["emotion"] == "🙂")]["text"].to_list()
happiness = pd.DataFrame({"content": h1 + h2 + h3 + h4, "emotion": 2})

neutral = tweet_emotions.loc[(tweet_emotions["sentiment"] == "neutral") | (tweet_emotions["sentiment"] == "boredom")]["content"].to_list()
neutral = pd.DataFrame({"content": neutral, "emotion": 3})

l1 = tweet_emotions.loc[(tweet_emotions["sentiment"] == "love")]["content"].to_list()
l2 = text.loc[(text["label"] == 2)]["text"].to_list()
love = pd.DataFrame({"content": l1 + l2, "emotion": 4})

angry.to_csv("./new_data/multiclass/angry.csv", index=False)
sadness.to_csv("./new_data/multiclass/sadness.csv", index=False)
happiness.to_csv("./new_data/multiclass/happiness.csv", index=False)
neutral.to_csv("./new_data/multiclass/neutral.csv", index=False)
love.to_csv("./new_data/multiclass/love.csv", index=False)