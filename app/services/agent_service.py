from app.config import GOOGLE_API_KEY
from app.services.vector_store import load_vector_store
from fastapi import Response, status
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
from langchain_community.tools import DuckDuckGoSearchRun, tool
from langchain.tools import StructuredTool
from pydantic import BaseModel, Field
from app.services.vector_store import get_retriever
from langchain_core.documents import Document
from langchain.prompts import PromptTemplate

# --- 1. Define Agent Tools ---


@tool
def search(query: str) -> str:
    """
    Use this tool to search the web when the user query requires external info.
    Format strictly:
    Action: search
    Action Input: <your query>
    """
    search_tool = DuckDuckGoSearchRun()
    results = search_tool.invoke(query)
    return results


class GetAllChunksInput(BaseModel):
    user_id: str = Field(..., description="Unique identifier of the user.")


def getAllChunks_fn(user_id: str) -> list[str]:
    print(f"🔍 [Tool] Fetching all chunks for user {user_id}...")
    vectorstore = load_vector_store(user_id)

    if vectorstore is None:
        return [
            "Error: No health report found for this user. Please upload a document first."
        ]

    try:
        # Access all documents from the docstore
        docs = [doc.page_content for doc in vectorstore.docstore._dict.values()]
        print(f"✅ [Tool] Retrieved {len(docs)} chunks for user {user_id}")
        return docs
    except Exception as e:
        print(f"❌ [Tool] Error while fetching chunks for user {user_id}: {e}")
        return [f"Error fetching data: {e}"]


getAllChunks = StructuredTool.from_function(
    func=getAllChunks_fn,
    name="getAllChunks",
    description=""""
Purpose:
Use this tool only when the user's question requires understanding the entire health report, not just a few sections or lab results.

When to use:

The user explicitly requests a summary, full analysis, or overall interpretation of their health report.

The query includes phrases like:
“Summarize my health report”, “Explain my complete blood test”, “Give me an overview”, or “Analyze my whole report”.

Or, if after retrieving partial chunks, the context seems insufficient to confidently answer.

When not to use:

If the retriever already provides enough relevant chunks to answer the user’s query accurately.

If the question focuses on a specific test, metric, or section (e.g., “What does my cholesterol level mean?”).
""",
    args_schema=GetAllChunksInput,
)

# --- 2. Initialize Agent ---

# llm
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-lite",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.2,
    # convert_system_message_to_human=True,  # Helps with some ReAct prompts
)

tools = [search, getAllChunks]

# Pull the standard ReAct prompt
prompt_template = hub.pull("hwchase17/react")

# Create the agent
agent = create_react_agent(llm=llm, prompt=prompt_template, tools=tools)

# Create the agent executor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,  # Logs agent "thinking" to console
    handle_parsing_errors=True,
)

# --- 3. Define the Agent's Input Prompt Template ---

# This prompt guides the agent on *how* to answer the user's query
PROMPT_TEMPLATE = PromptTemplate(
    template="""
You are given a user's personal health report data.

User ID: {user_id}
User Query: {query}

Health Data:
{data}

Your task:
1. Analyze whether each reported value is within or outside its normal range.
2. Respond naturally and helpfully:
   - ✅ Within range → Give a reassuring message and include the stats.
   - ⚠️ Outside range → Give a kind, short explanation of possible causes and one-line advice or precaution.
3. Adapt your tone to sound caring, knowledgeable, and clear — like a friendly health advisor.
4. Keep your explanation concise and professional.
5. Include the numeric stats and normal ranges where relevant.
""",
    input_variables=["data", "query", "user_id"],
)

# --- 4. Main Query Function ---


def run_agent_query(user_id: str, query: str) -> dict:
    """
    Runs the full RAG-then-Agent workflow:
    1. Fetches the user's retriever.
    2. Gets relevant docs.
    3. Formats a prompt with the docs.
    4. Invokes the agent with the rich prompt.
    """

    print(f"--- Starting new query for {user_id} ---")

    # Step 1: Load the retriever (as requested from Cell 114)
    retriever = get_retriever(user_id)
    if not retriever:
        return {
            "code": status.HTTP_404_NOT_FOUND,
            "message": f"I'm sorry, but I couldn't find a health report for user {user_id}. Please upload one first.",
        }

    print(f"✅ Retriever loaded for {user_id}")

    # Step 2: Fetch relevant docs (as requested from Cell 115)
    try:
        docs: list[Document] = retriever.invoke(query)
        if not docs:
            print(
                "⚠️ No relevant documents found by retriever, agent will have to rely on tools."
            )
            fetched_data = ["No specific data found for this query."]
        else:
            # (From Cell 121)
            fetched_data = [doc.page_content for doc in docs]
            print(f"✅ Retriever found {len(fetched_data)} relevant chunks.")

    except Exception as e:
        print(f"❌ Error during retrieval: {e}")
        return {
            "code": status.HTTP_500_INTERNAL_SERVER_ERROR,
            "message": "I'm sorry, I encountered an error while retrieving your health data.",
        }

    # Step 3: Format the prompt with fetched data (as requested from Cell 123)
    try:
        agent_input_prompt = PROMPT_TEMPLATE.invoke(
            {
                "data": fetched_data,
                "query": query,
                "user_id": user_id,
            }
        )

        print(f"✅ Prompt formatted for agent.")

    except Exception as e:
        print(f"❌ Error formatting prompt: {e}")
        return {
            "code": status.HTTP_500_INTERNAL_SERVER_ERROR,
            "message": "I'm sorry, I encountered an error while preparing your query.",
        }

    # Step 4: Invoke the agent with the RAG-filled prompt (as requested)
    try:
        response = agent_executor.invoke(
            {"input": agent_input_prompt.to_string()}  # Pass the formatted string
        )

        print(f"✅ Agent execution complete.")
        return {
            "code": status.HTTP_200_OK,
            "message": response["output"],
        }

    except Exception as e:
        print(f"❌ Error during agent execution: {e}")
        return {
            "code": status.HTTP_500_INTERNAL_SERVER_ERROR,
            "message": f"I'm sorry, I encountered an error while processing your request : {e}.",
        }
