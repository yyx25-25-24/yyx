from langchain import agents
from langchain.tools import tool
from langchain_community.llms import Tongyi

def echo(query: str) -> str:
    """Echo the query back."""
    return 'ECHO:' + query

echo_tool = tool(echo)
llm = Tongyi(model_name='qwen-turbo', temperature=0, api_key='sk-3c471a21f2df4a04a0228d5d1a001234')
agent = agents.create_agent(model=llm, tools=[echo_tool], system_prompt='You are helpful.')
print(type(agent))
print(hasattr(agent, 'run'))
print(hasattr(agent, 'invoke'))
print([m for m in dir(agent) if m in ('run','invoke','__call__')])
