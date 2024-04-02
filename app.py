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
    "had lunch?",
    "how was day?",
    "who is you friend",
    "what work do you do",
    "how old are you",
    "whats your name?",
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
    "how's it going?"
]

train_labels = [
    "Hi",
    "I'm fine, how about you?",
    "good to hear",
    "Good morning to you",
    "Good evening, how can I help you?",
    "Nice to meet you too",
    "Not much, just hanging out",
    "It's going well, thank you",
    "Hello!",
    "Good afternoon to you too",
    "I'm here to assist you",
    "Likewise!",
    "Yes, I have a question",
    "no",
    "It was great, thanks for asking!",
    "I'm a chatbot, here to assist you!",
    "Why don't we hear about the scarecrow's award?",
    "I like to chat with users like you!",
    "I wish I had a pet, but I'm just a program.",
    "I recommend 'The Alchemist' by Paulo Coelho.",
    "It's sunny today with a high of 75°F.",
    "I'm a chatbot, so I don't have favorite movies.",
    "I exist in the realm of code.",
    "I'm not capable of eating, but I like the idea of pizza!",
    "I'm not sure about aliens, but I'm here to chat with you!",
    "I like all colors equally!",
    "I'm not into sports, but I enjoy conversations.",
    "I spend my time chatting with users like you!",
    "Being a helpful chatbot is my dream job!",
    "i am not a human",
    "good",
    "i dont have friends",
    " i answer people's question",
    "i dont have age",
    "i am a chatbot",
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
    "i'm doing well. how about you?"
]
# Additional training data
additional_train_data = [
    "Hey there!",
    "How's it going?",
    "I'm doing well, thank you",
    "Good to see you",
    "What's new?",
    "I'm just chilling",
    "Any plans for the weekend?",
    "Not much happening here",
    "How's work/school?",
    "Nice weather we're having",
]

# Additional training labels corresponding to the new training data
additional_train_labels = [
    "Hey!",
    "It's going great, thanks for asking!",
    "That's good to hear",
    "Good to see you too!",
    "Not much, just the usual",
    "Same here",
    "Not really, just relaxing",
    "Same here, quiet day",
    "It's going well, thanks for asking",
    "Yes, it's lovely outside",
]

# Append additional training data and labels to the existing ones
train_data.extend(additional_train_data)
train_labels.extend(additional_train_labels)



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
