import pandas as pd

tweet_emotions = pd.read_csv("./data/tweet_emotions.csv")
emotion_angry = pd.read_csv("./data/Emotion(angry).csv")["content"].to_list()
emotion_happy = pd.read_csv("./data/Emotion(happy).csv")["content"].to_list()
emotion_sad = pd.read_csv("./data/Emotion(sad).csv")["content"].to_list()
text = pd.read_csv("./data/text.csv")
text_emotion = pd.read_csv("./data/Text_Emotion.csv")

print(tweet_emotions["sentiment"].unique())
en1 = tweet_emotions.loc[(tweet_emotions["sentiment"] == "empty") | (tweet_emotions["sentiment"] == "sadness") | (tweet_emotions["sentiment"] == "worry") | (tweet_emotions["sentiment"] == "anger") | (tweet_emotions["sentiment"] == "hate")]["content"].to_list()
en2 = emotion_angry + emotion_sad
en3 = text.loc[(text["label"] == 0) | (text["label"] == 3) | (text["label"] == 4)]["text"].to_list()
en4 = text_emotion[(text_emotion["emotion"] == "☹️")]["text"].to_list()

negative = pd.DataFrame({"content": en1 + en2 + en3 + en4, "emotion": 0})

ep1 = tweet_emotions.loc[(tweet_emotions["sentiment"] == "enthusiasm") | (tweet_emotions["sentiment"] == "love") | (tweet_emotions["sentiment"] == "fun") | (tweet_emotions["sentiment"] == "happiness") | (tweet_emotions["sentiment"] == "relief") | (tweet_emotions["sentiment"] == "fun")]["content"].to_list()
ep2 = emotion_happy
ep3 = text.loc[(text["label"] == 1) | (text["label"] == 2)]["text"].to_list()
ep4 = text_emotion[(text_emotion["emotion"] == "🙂")]["text"].to_list()

positive =  pd.DataFrame({"content": ep1 + ep2 + ep3 + ep4, "emotion": 1})

negative.to_csv("./new_data/tri/negative.csv", index=False)
positive.to_csv("./new_data/tri/positive.csv", index=False)

neutral = tweet_emotions.loc[(tweet_emotions["sentiment"] == "neutral") | (tweet_emotions["sentiment"] == "boredom")]["content"].to_list()
neutral = pd.DataFrame({"content": neutral, "emotion": 2})
neutral.to_csv("./new_data/tri/neutral.csv", index=False)