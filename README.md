LangGraph Agent Project

Installation
To install all required libraries, run the following command in your terminal:
**pip install -r requirements.txt**

Running the Server
Start the server using Uvicorn with the command below:
**uvicorn mcp_server:app --host 0.0.0.0 --port 8000**

Starting the Agent
Once the server is running, execute the following command to start the agent process:
**python agent.py**
