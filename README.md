# ChatBot with Sentiment Analysis and Music Recommendation

## Standard-Readme Compliant

A chatbot application integrated with sentiment analysis and a music recommendation system.

## Table of Contents
- [Background](#background)
- [Install](#install)
- [Usage](#usage)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Background
This project is a chatbot system that combines **natural language processing (NLP), sentiment analysis, and music recommendations** to create an engaging user experience. The chatbot can:
- **Understand and classify user inputs** using a trained neural network.
- **Analyze sentiment** using IBM Watson’s Tone Analyzer.
- **Recommend songs** based on the detected sentiment using the Last.fm API.

## Install
This project requires Python and some dependencies. To install them, run:

```sh
pip install anvil-uplink tensorflow keras nltk ibm-watson requests
```

## Usage
To start the chatbot, run the Python script:

```sh
python chat_bot_recommender.py
```

The chatbot will prompt you for user input and respond based on its trained model. It will also analyze recent messages for sentiment and provide music recommendations.

## Features
- **Conversational Chatbot**: Responds to user queries based on trained intents.
- **Sentiment Analysis**: Detects emotions in user input.
- **Music Recommendations**: Suggests songs based on the detected emotion.
- **Machine Learning-Based NLP**: Uses a neural network for intent classification.

## Technologies Used
- **Machine Learning**: TensorFlow, Keras
- **NLP**: NLTK (tokenization, lemmatization)
- **APIs**: IBM Watson Tone Analyzer, Last.fm API
- **Deployment**: Anvil Uplink

## Project Structure
```
chatbot_project/
│── chatbot_model.h5       # Trained model
│── intents.json           # Intents and responses
│── chatbot_recommender.py # Main chatbot script
│── words.pkl              # Processed words
│── classes.pkl            # Processed classes
│── requirements.txt       # Required dependencies
```

## Contributing
Feel free to open an issue or submit a pull request if you have suggestions or improvements.

## License
This project is licensed under the MIT License.

