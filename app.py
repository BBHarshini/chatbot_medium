import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
import streamlit as st

# Step 1: Preparing the Training Data
train_data = [
    "Hello",
    "How are you?",
    "good",
    "Good morning",
    "Good evening",
    "Nice to meet you",
    "What's up?",
    "How's your day going?",
    "Greetings!",
    "Good afternoon",
    "How can I assist you?",
    "Pleasure to see you",
    "Is there anything I can help with?",
    "you know maths?",
    "How was your weekend?",
    "Are you a human or a bot?",
    "Tell me a joke.",
    "What do you like to do for fun?",
    "Do you have any pets?",
    "Can you recommend a good book?",
    "What's the weather like today?",
    "What's your favorite movie?",
    "Where are you from?",
    "What's your favorite food?",
    "Do you believe in aliens?",
    "What's your favorite color?",
    "Do you like sports?",
    "How do you spend your free time?",
    "What's your dream job?",
    "hi, how are you doing?",
    "i'm fine. how about yourself?",
    "i'm pretty good. thanks for asking.",
    "no problem. so how have you been?",
    "i've been great. what about you?",
    "i've been good. i'm in school right now.",
    "what school do you go to?",
    "i go to pcc.",
    "do you like it there?",
    "it's okay. it's a really big campus.",
    "good luck with school.",
    "thank you very much.",
    "how's it going?",
    "i'm doing well. how about you?",
    "never better, thanks.",
    "so how have you been lately?",
    "i've actually been pretty good. you?",
    "i'm actually in school right now.",
    "which school do you attend?",
    "i'm attending pcc right now.",
    "are you enjoying it there?",
    "it's not bad. there are a lot of people there.",
    "good luck with that.",
    "thanks.",
    "how are you doing today?",
    "i'm doing great. what about you?",
    "i'm absolutely lovely, thank you.",
    "everything's been good with you?",
    "i haven't been better. how about yourself?",
    "i started school recently.",
    "where are you going to school?",
    "i'm going to pcc.",
    "how do you like it so far?",
    "i like it so far. my classes are pretty good right now.",
    "i wish you luck.",
    "it's an ugly day today.",
    "i know. i think it may rain.",
    "it's the middle of summer, it shouldn't rain today.",
    "that would be weird.",
    "yeah, especially since it's ninety degrees outside.",
    "i know, it would be horrible if it rained and it was hot outside.",
    "yes, it would be.",
    "i really wish it wasn't so hot every day.",
    "me too. i can't wait until winter.",
    "winter is great. i wish it didn't get so cold sometimes though.",
    "i would rather deal with the winter than the summer.",
    "it's such a nice day.",
    "yes, it is.",
    "it looks like it may rain soon.",
    "yes, and i hope that it does.",
    "why is that?",
    "i really love how rain clears the air.",
    "me too. it always smells so fresh after it rains.",
    "i really hope it rains today.",
    "yeah, me too.",
    "isn't it a nice day?",
    "yes, i think so too.",
    "it seems that it may rain today.",
    "hopefully it will.",
    "how come?",
    "i like how clear the sky gets after it rains.",
    "i feel the same way. it smells so good after it rains.",
    "i especially love the night air when it rains.",
    "really? why?",
    "because you can see the stars perfectly.",
    "i really want it to rain today.",
    "yeah, so do i.",
    "don't you think it's nice out?",
    "yes, i think so too.",
    "i think that it's going to rain.",
    "i hope that it does rain.",
    "you like the rain?",
    "the sky looks so clean after it rains. i love it.",
    "i understand. rain does make it smell cleaner.",
    "i love most how it is at night after it rains.",
    "how come?",
    "you can see the stars so much more clearly after it rains.",
    "i really want it to rain today.",
    "yeah, so do i.",
    "isn't it a nice day?",
    "yes, i think so too.",
    "it looks like it may rain today.",
    "yes, and i hope that it does.",
    "why is that?",
    "i really love how rain clears the air.",
    "me too. it always smells so fresh after it rains.",
    "i really hope it rains today.",
    "yeah, me too.",
    "isn't it a nice day?",
    "yes, i think so too.",
    "it seems that it may rain today.",
    "hopefully it will.",
    "how come?",
    "i like how clear the sky gets after it rains.",
    "i feel the same way. it smells so good after it rains.",
    "i especially love the night air when it rains.",
    "really? why?",
    "because you can see the stars perfectly.",
    "i really want it to rain today.",
    "yeah, so do i.",
    "don't you think it's nice out?",
    "yes, i think so too.",
    "i think that it's going to rain.",
    "i hope that it does rain.",
    "you like the rain?",
    "the sky looks so clean after it rains. i love it.",
    "i understand. rain does make it smell cleaner.",
    "i love most how it is at night after it rains.",
    "how come?",
    "you can see the stars so much more clearly after it rains.",
    "i really want it to rain today.",
    "yeah, so do i.",
    "isn't it a nice day?",
    "yes, i think so too.",
    "it looks like it may rain today.",
    "yes, and i hope that it does.",
    "why is that?",
    "i really love how rain clears the air.",
    "me too. it always smells so fresh after it rains.",
    "i really hope it rains today.",
    "yeah, me too.",
    "isn't it a nice day?",
    "yes, i think so too.",
    "it seems that it may rain today.",
    "hopefully it will.",
    "how come?",
    "i like how clear the sky gets after it rains.",
    "i feel the same way. it smells so good after it rains.",
    "i especially love the night air when it rains."
]


test_data = [
    "Hi, how are you today?",
    "What's your favorite color?",
    "Do you enjoy reading?",
    "How's the weather where you are?",
    "Tell me about your hobbies.",
    "What's your opinion on current events?",
    "Have you watched any good movies lately?",
    "Do you prefer coffee or tea?",
    "What's your favorite season?",
    "Are you more of a morning person or a night owl?",
    "Tell me about your family.",
    "What's your favorite cuisine?",
    "Do you like to travel?",
    "What's your idea of a perfect day?",
    "Tell me about your dreams for the future.",
    "Do you believe in luck?",
    "What's your favorite holiday?",
    "Are you an introvert or an extrovert?",
    "What's your biggest fear?",
    "Tell me about a memorable experience you've had.",
    "What's your favorite type of music?",
    "Do you enjoy cooking?",
    "What's your opinion on social media?",
    "Tell me about your best friend.",
    "What's the most adventurous thing you've ever done?",
    "What's your favorite dessert?",
    "Do you like to exercise?",
    "What's your opinion on climate change?",
    "Tell me about your childhood.",
    "Do you have any siblings?",
    "What's your favorite way to relax?",
    "Are you a morning person or a night owl?",
    "What's your favorite type of cuisine?",
    "Do you enjoy outdoor activities?",
    "What's your favorite book?",
    "Tell me about a challenge you've overcome.",
    "Do you like to attend live events?",
    "What's your opinion on technology?",
    "Tell me about your education.",
    "What's your favorite TV show?",
    "Do you believe in fate?",
    "What's your favorite type of movie?",
    "Tell me about your pet, if you have one.",
    "What's your favorite place you've visited?",
    "Do you like to dance?",
    "What's your opinion on art?",
    "Tell me about a goal you've achieved.",
    "What's your favorite sport?",
    "Do you have any regrets?",
    "What's your favorite quote?",
    "Tell me about a role model you admire.",
    "What's your favorite restaurant?",
    "Do you like to attend parties?",
    "What's your opinion on education?",
    "Tell me about a place you'd like to visit.",
    "What's your favorite hobby?",
    "Do you like animals?",
    "What's your opinion on social issues?",
    "Tell me about your job or career.",
    "What's your favorite type of food?",
    "Do you like to listen to music?",
    "What's your opinion on relationships?",
    "Tell me about a dream you've had.",
    "What's your favorite holiday destination?",
    "Do you like to go shopping?",
    "What's your opinion on fashion?",
    "Tell me about a lesson you've learned.",
    "What's your favorite way to spend a weekend?",
    "Do you like to watch sports?",
    "What's your opinion on politics?",
    "Tell me about a favorite memory from childhood.",
    "What's your favorite place to relax?",
    "Do you like to travel solo or with company?",
    "What's your opinion on privacy?",
    "Tell me about a time you took a risk.",
    "What's your favorite thing to do with friends?",
    "Do you like to try new things?",
    "What's your opinion on the environment?",
    "Tell me about a favorite family tradition.",
    "What's your favorite way to stay active?",
    "Do you enjoy attending concerts?",
    "What's your opinion on work-life balance?",
    "Tell me about a book that changed your life.",
    "What's your favorite outdoor activity?",
    "Do you like to visit museums?",
    "What's your opinion on social norms?",
    "Tell me about a goal you're currently working towards.",
    "What's your favorite thing to do on a rainy day?",
    "Do you enjoy gardening?",
    "What's your opinion on religion?",
    "Tell me about a difficult decision you've made.",
    "What's your favorite thing about yourself?",
    "Do you like to cook or bake?",
    "What's your opinion on artificial intelligence?",
    "Tell me about a favorite childhood toy.",
    "What's your favorite way to unwind after a long day?",
    "Do you enjoy going to the theater?",
    "What's your opinion on mental health?",
    "Tell me about a time you made someone smile.",
    "What's your favorite type of cuisine to cook?",
    "Do you like to go hiking?",
    "What's your opinion on online dating?",
    "Tell me about a place you'd like to live someday.",
    "What's your favorite type of weather?",
    "Do you enjoy going to the beach?",
    "What's your opinion on social justice?",
    "Tell me about a time you made a mistake.",
    "What's your favorite type of exercise?",
    "Do you like to attend cultural events?",
    "What's your opinion on peer pressure?",
    "Tell me about a goal you have for the future.",
    "What's your favorite thing to do on a sunny day?",
    "Do you enjoy spending time outdoors?",
    "What's your opinion on volunteering?",
    "Tell me about a time you helped someone in need.",
    "What's your favorite type of cuisine to eat out?",
    "Do you like to go biking?",
    "What's your opinion on the importance of family?",
    "Tell me about a time you felt proud of yourself.",
    "What's your favorite thing to do on a lazy day?",
    "Do you enjoy going to the zoo?",
    "What's your opinion on cultural diversity?",
    "Tell me about a dream you have for the future.",
    "What's your favorite type of music to listen to?",
    "Do you like to go camping?",
    "What's your opinion on self-care?",
    "Tell me about a time you stood up for what you believe in.",
    "What's your favorite thing to do with your family?",
    "Do you enjoy going to amusement parks?",
    "What's your opinion on the power of positivity?",
    "Tell me about a time you faced a challenge head-on.",
    "What's your favorite way to spend time with loved ones?",
    "Do you like to go on road trips?",
    "What's your opinion on stress management?",
    "Tell me about a time you learned something new.",
    "What's your favorite type of movie to watch with friends?",
    "Do you enjoy going to the gym?",
    "What's your opinion on time management?",
    "Tell me about a time you felt grateful.",
    "What's your favorite thing to do with your significant other?",
    "Do you like to go skiing?",
    "What's your opinion on work ethic?",
    "Tell me about a time you felt inspired.",
    "What's your favorite type of food to order takeout?",
    "Do you enjoy going to the aquarium?",
    "What's your opinion on the power of perseverance?",
    "Tell me about a time you achieved something against all"
]


# Step 2: Data Preprocessing
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(train_labels)

tokenizer = keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(train_data)
train_sequences = tokenizer.texts_to_sequences(train_data)
train_sequences = keras.preprocessing.sequence.pad_sequences(train_sequences)

# Step 3: Building and Training the Chatbot Model
model = keras.models.Sequential()

model.add(keras.layers.Embedding(len(tokenizer.word_index) + 1, 100))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dense(len(train_labels), activation='softmax'))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(train_sequences, encoded_labels, epochs=50)

# Step 4: Generating Responses
def generate_response(text):
    sequence = tokenizer.texts_to_sequences([text])
    sequence = keras.preprocessing.sequence.pad_sequences(sequence, maxlen=train_sequences.shape[1])
    prediction = model.predict(sequence)
    predicted_label = np.argmax(prediction)
    response = label_encoder.inverse_transform([predicted_label])[0]
    return response

# Streamlit UI
st.title("Simple Chatbot")

user_input = st.text_input("Enter a message:")
if user_input:
    response = generate_response(user_input)
    st.write("ChatBot:", response)
