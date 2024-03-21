Promptior Chatbot
This chatbot was developed as part of the AI Engineer assessment process for Promptior company. It is a chatbot based on the langchain library that utilizes the langserve API. It responds with all the requested information based on the data stored in data.txt. The main files for this chatbot are serve.py and chain.py.

Installation
Clone this repository to your local machine using the following command:
git clone <repository URL>

Install the required dependencies by running the following command:
langchain
langchain_openai
faiss-cpu
langcorn

Used template: 
pirate-speak from langchain repository

Usage
To start the chatbot, run the serve.py file. Make sure you have the data.txt file present in the same directory so the chatbot can access the necessary information to respond to questions.

Main Files
serve.py: This file contains the main code to run the chatbot.
chain.py: Here is the implementation of the chatbot logic using the langchain library.

Contributions
Contributions are welcome! If you would like to contribute to this project, please create a pull request describing the proposed changes.

Contact
If you have any questions or suggestions, feel free to contact the development team at cardlean94@gmail.com.