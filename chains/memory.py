# centralized_memory.py
from langchain.memory import ConversationBufferMemory

# Centralized memory instance
central_memory = ConversationBufferMemory(return_messages=True)
