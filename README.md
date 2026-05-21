## Design and Implementation of a Multidocument Retrieval Agent Using LlamaIndex

### AIM:
To design and implement a multidocument retrieval agent using LlamaIndex to extract and synthesize information from multiple research articles, and to evaluate its performance by testing it with diverse queries, analyzing its ability to deliver concise, relevant, and accurate responses.

### PROBLEM STATEMENT:

Traditional search engines and basic retrieval systems struggle to synthesize information spread across multiple distinct documents or research papers. When faced with complex queries that require connecting insights from different sources (such as evaluating a methodology in one paper and comparing it with results in another), standard keyword search fails. This project addresses the challenge of building an intelligent, LLM-powered multidocument retrieval agent using LlamaIndex. The system must automatically ingest multiple PDF research papers, build individual vector and summary tools for each document, and leverage an agentic framework to dynamically reason, select the appropriate tools, and provide comprehensive, context-aware answers to complex analytical queries.

### DESIGN STEPS:

### STEP 1: Environment Setup and Tool Initialization
Install necessary libraries (llamaindex, openai, nest_asyncio) and securely configure the OpenAI API key.

Apply nest_asyncio to allow nested asynchronous loops, which are critical for running concurrent document operations.

Define a helper utility function (get_doc_tools) that takes a document path and its identity to automatically generate two distinct query tools per paper: a Vector Tool (for specific, granular semantic search queries) and a Summary Tool (for high-level, document-wide synthesis).

### STEP 2: Document Processing and Agent Assembly
Iterate through the specified research papers (.pdf files), creating the vector and summary tool pairs for each, and store them in a dictionary mapping.

Flatten the complete collection of tools into a single, unified list (initial_tools) that the agent can access.

Initialize the core Large Language Model (GPT-3.5-Turbo) to act as the reasoning engine.

Construct a FunctionCallingAgentWorker from the compiled toolset and wrap it with an AgentRunner to orchestrate multi-step reasoning, tool selection, and response execution loop.

### STEP 3: Query Execution and Performance Evaluation
Formulate multi-hop, analytical test queries that require the agent to search within a specific paper (e.g., analyzing teacher data visualization limitations) or summarize specific contexts under a strict word limit (e.g., summarizing Travis Weiland's paper).

Execute the queries using the agent's .query() method while setting verbose=True to observe the agent's internal thought process and selection of tools.

Capture, print, and analyze the final output responses to evaluate the agent's correctness, concise phrasing, and multi-source accuracy.
### PROGRAM:

~~~
# Setup and API Keys
from helper import get_openai_api_key
OPENAI_API_KEY = get_openai_api_key()

import nest_asyncio
nest_asyncio.apply()

# Define document URLs and filenames
urls = [
    "https://openreview.net/pdf?id=VtmBAGCN7o",
    "https://openreview.net/pdf?id=6PmJoRfdaK",
    "https://openreview.net/pdf?id=hSyW5go0v8",
]

papers = [
    "14_RESEARCH_TO_PRACTICE_DESIGN.pdf",
    "18_GENERATION_AI_PEDAGOGICAL_F.pdf",
    "24_REFACTORING_COMPUTER_SCIENC.pdf",
]

# Initialize document tools
from utils import get_doc_tools
from pathlib import Path

paper_to_tools_dict = {}
for paper in papers:
    print(f"Getting tools for paper: {paper}")
    vector_tool, summary_tool = get_doc_tools(paper, Path(paper).stem)
    paper_to_tools_dict[paper] = [vector_tool, summary_tool]

# Flatten tools list for the agent
initial_tools = [t for paper in papers for t in paper_to_tools_dict[paper]]

# Initialize LLM and Agent
from llama_index.llms.openai import OpenAI
from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.agent import AgentRunner

llm = OpenAI(model="gpt-3.5-turbo")

agent_worker = FunctionCallingAgentWorker.from_tools(initial_tools, llm=llm, verbose=True)
agent = AgentRunner(agent_worker)

# Query 1: Analysis and Evaluation
response1 = agent.query(
    "According to the pilot study findings, what were the two main limitations the author discovered "
    "when applying their initial one-dimensional framework to analyze how teachers read data visualizations? "
    "and then tell me about the evaluation results"
)
print(str(response1))

# Query 2: Document Summary
response2 = agent.query("Summarize the attached research paper by Travis Weiland in under 250 words")
print(str(response2))
~~~

### OUTPUT:

<img width="1429" height="762" alt="image" src="https://github.com/user-attachments/assets/7b84c185-6a0b-4629-9496-b6b316f899e2" />

<img width="1523" height="791" alt="image" src="https://github.com/user-attachments/assets/af5ad2fd-8129-41bb-b41c-1bbcd44c14c8" />



### RESULT:
Thus the program has been executed successfully
