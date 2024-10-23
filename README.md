# superchatbot

# Chat with Assistant

This is a Streamlit application that allows users to interact with a Groq-powered AI assistant. The app fetches trainer data from a Supabase database, generates embeddings using the SentenceTransformer model, and allows users to search for trainers based on their queries.

## Features

- Chat with an AI assistant powered by Groq.
- Fetch trainer data from Supabase.
- Generate embeddings for trainer descriptions.
- Perform similarity searches to find related trainers.
- Clickable buttons to initiate chats with trainers.

## Technologies Used

- [Streamlit](https://streamlit.io/) - Framework for building web applications.
- [Groq](https://groq.com/) - AI model for generating responses.
- [Supabase](https://supabase.io/) - Backend as a service for database management.
- [Sentence Transformers](https://www.sbert.net/) - For generating embeddings from text.
- [FAISS](https://faiss.ai/) - A library for efficient similarity search and clustering of dense vectors.
- [dotenv](https://pypi.org/project/python-dotenv/) - For loading environment variables from a `.env` file.

## Installation

To run this application locally, follow these steps:

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/chat-with-groq-assistant.git
   cd chat-with-groq-assistant

Create a virtual environment (optional but recommended):
bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

Install the required packages:
bash
pip install -r requirements.txt

Set up environment variables:
Create a .env file in the root directory of the project and add the following lines:
text
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
GROQ_API_KEY=your_groq_api_key

Run the application:
bash
streamlit run app.py
