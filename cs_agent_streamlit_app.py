import sys
import pysqlite3
sys.modules["sqlite3"] = pysqlite3
import sqlite3
import streamlit as st
import pandas as pd
import google.generativeai as genai
from google.api_core import retry
import os
from dotenv import load_dotenv

# Load variables from .env
load_dotenv()

# Gemini Configuration
api_key = os.getenv('GOOGLE_API_KEY')
if not api_key:
    st.error("GOOGLE_API_KEY não encontrada. Por favor, configure a chave API nas configurações do Streamlit Cloud.")
    st.stop()

genai.configure(api_key=api_key)
retry_policy = {"retry": retry.Retry(predicate=retry.if_transient_error)}

## -- Database connection -- ##
def init_db_connection():
    try:
        print("Tentando conectar ao banco de dados...")
        db_conn = sqlite3.connect('my_database.db')
        print("Conexão com banco de dados estabelecida com sucesso!")
        return db_conn
    except Exception as e:
        print(f"Erro ao conectar ao banco de dados: {e}")
        st.error(f"Erro de conexão com o banco: {e}")
        return None

db_conn = init_db_connection()

def list_tables() -> list[str]:
    """Retrieve the names of all tables in the database."""
    # Include print logging statements so you can see when functions are being called.
    print(' - DB CALL: list_tables')

    cursor = db_conn.cursor()

    # Fetch the table names.
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")

    tables = cursor.fetchall()
    return [t[0] for t in tables]

list_tables()

def describe_table(table_name: str) -> list[tuple[str, str]]:
    """Look up the table schema.

    Returns:
      List of columns, where each entry is a tuple of (column, type).
    """
    print(' - DB CALL: describe_table')

    cursor = db_conn.cursor()

    cursor.execute(f"PRAGMA table_info({table_name});")

    schema = cursor.fetchall()
    # [column index, column name, column type, ...]
    return [(col[1], col[2]) for col in schema]


describe_table("produtos_tec")

## -- Database functions -- ##
def execute_query(sql: str) -> list[list[str]]:
    """Execute a SELECT statement, returning the results."""
    print(' - DB CALL: execute_query')
    print(f' - SQL Query before: {sql}')
    
    if db_conn is None:
        print("Erro: Conexão com banco de dados não estabelecida")
        return []
        
    # Remove escaped quotes
    sql = sql.replace("\\'", "'")
    print(f' - SQL Query after: {sql}')

    if not sql or not isinstance(sql, str):
        raise ValueError("SQL query must be a non-empty string")

    cursor = db_conn.cursor()
    
    try:
        cursor.execute(sql)
        results = cursor.fetchall()
        print(f' - Query results: {results}')
        return results
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        st.error(f"Erro na query: {e}")
        return []
        
## -- Model configuration -- ##
# These are the Python functions defined above.

def init_model():
    SYSTEM_INSTRUCTION = """You are a helpful Customer Service Agent for a computer store. You will take the users questions and turn them into SQL queries using the tools
    available. 

    DATABASE INFORMATION:
    - Table name: produtos_tec
    - Columns: id, nome, descricao, preco, estoque, categoria

    IMPORTANT SQL RULES:
    - Always use single quotes (') for string literals
    - DO NOT escape quotes in the SQL query
    - For category searches use: WHERE LOWER(categoria) = 'mouse'
    - For partial matches use: WHERE LOWER(nome) LIKE '%mouse%'
    - Use proper SQLite syntax

    EXAMPLE QUERIES:
    - Correct: SELECT * FROM produtos_tec WHERE LOWER(categoria) = 'mouse'
    - Correct: SELECT * FROM produtos_tec WHERE preco < 1000
    - Correct: SELECT * FROM produtos_tec WHERE LOWER(nome) LIKE '%mouse%'

    When responding to users:
    1. Generate appropriate SQL queries using the rules above
    2. Use the query results to provide helpful, customer-friendly responses
    3. Include relevant product details in your answers
    """
    
    db_tools = [list_tables, describe_table, execute_query]

    model = genai.GenerativeModel(
        "models/gemini-1.5-flash-latest", 
        tools=db_tools, 
        system_instruction=SYSTEM_INSTRUCTION
    )
    
    return model

## -- Streamlit page configuration -- ##
st.set_page_config(
    page_title="Computer Store Assistant",
    page_icon="💻",
    layout="wide"
)

# Add examples of questions before the chat input
example_categories = {
    "💰 Price-related Questions": [
        "What are the cheapest products?",
        "Show me all products under $500",
        "What's the most expensive notebook available?",
        "Show me products between $1000 and $2000",
    ],
    "📁 Category-based Questions": [
        "What gaming mice do you have?",
        "Show me all available notebooks",
        "What wireless keyboards are in stock?",
        "List all products in the mouse category",
    ],
    "📦 Stock-related Questions": [
        "Which products are currently in stock?",
        "Do you have any gaming keyboards available?",
        "Show me notebooks that are in stock",
    ],
    "🔍 Product Search Questions": [
        "Do you have any Logitech products?",
        "Show me all RGB keyboards",
        "Are there any mechanical keyboards?",
    ],
    "🎯 Combined Criteria Questions": [
        "What are the cheapest gaming mice in stock?",
        "Show me wireless keyboards under $100",
        "What are the most expensive gaming products?",
        "Do you have any budget notebooks in stock?"
    ]
}

## Main interface
def main():
    st.title("🤖 Virtual Assistant - Computer Store")

    model = init_model()
    
    chat = model.start_chat(
        enable_automatic_function_calling=True
        )
    
    # Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
        
    #User input
    if prompt := st.chat_input("Ask me anything about the computer store!"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate a response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    print("Enviando mensagem para o modelo...")
                    response = chat.send_message(prompt, request_options=retry_policy)
                    print("Resposta recebida do modelo")
                    if response.text:
                        print(f"Resposta: {response.text}")
                        st.session_state.messages.append({"role": "assistant", "content": response.text})
                        st.markdown(response.text)
                    else:
                        print("Resposta vazia do modelo")
                        error_message = "I apologize, but I couldn't generate a proper response. Please try rephrasing your question."
                        st.session_state.messages.append({"role": "assistant", "content": error_message})
                        st.error(error_message)
                except Exception as e:
                    print(f"Erro detalhado: {str(e)}")
                    error_message = f"An error occurred: {str(e)}"
                    st.session_state.messages.append({"role": "assistant", "content": error_message})
                    st.error(error_message)

    # Sidebar for examples
    # Enhanced sidebar
    with st.sidebar:
                
        st.title("🤖 Virtual Assistant - Computer Store")
        st.write("Ask me anything about the computer store!")
        
        # Add clear chat button
        if st.button("🗑️ Clear Chat", key="clear_chat_sidebar", use_container_width=True):
            print("Botão clicado!")  # Debug print
            st.session_state.messages = []
            st.rerun()       
        
        st.divider()
        
        st.subheader("📝 Example Questions")
        # Create expandable sections for each category
        for category, questions in example_categories.items():
            with st.expander(category):
                for question in questions:
                    if st.button(question, key=question, use_container_width=True):
                        # Send the selected question as input
                        st.session_state.messages.append({"role": "user", "content": question})
                        with st.chat_message("user"):
                            st.markdown(question)
                        
                        # Generate response for the selected question
                        with st.chat_message("assistant"):
                            with st.spinner("Thinking..."):
                                try:
                                    response = chat.send_message(question, request_options=retry_policy)
                                    if response.text:
                                        st.session_state.messages.append({"role": "assistant", "content": response.text})
                                        st.markdown(response.text)
                                    else:
                                        error_message = "I apologize, but I couldn't generate a proper response. Please try rephrasing your question."
                                        st.session_state.messages.append({"role": "assistant", "content": error_message})
                                        st.error(error_message)
                                except Exception as e:
                                    error_message = f"An error occurred: {str(e)}"
                                    st.session_state.messages.append({"role": "assistant", "content": error_message})
                                    st.error(error_message)
                                    print(f"Error details: {e}")
                        
                        # Rerun the app to show the new messages
                        st.rerun()

if __name__ == "__main__":
    main()
