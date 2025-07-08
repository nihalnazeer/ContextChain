v1.1
contextchain/
├── app/
│   ├── api/
│   │   └── server.py                 # FastAPI server for frontend queries
│   ├── engine/
│   │   ├── executor.py               # Executes tasks, integrates LangChain chains/agents
│   │   ├── validator.py              # Validates schemas, adds MCP validation
│   │   ├── trigger.py                # Handles triggers for task execution
│   │   └── langchain/
│   │       ├── chains.py             # Defines LangChain chains (e.g., RetrievalQA)
│   │       ├── agents.py             # Defines LangChain/OpenAI agents with tool-calling
│   │       └── tools.py              # Custom LangChain tools for ContextChain
│   ├── registry/
│   │   ├── version_manager.py        # Manages schema versioning
│   │   └── schema_loader.py          # Loads schemas, supports MCP metadata
│   ├── db/
│   │   ├── mongo_client.py           # MongoDB operations (local/Atlas)
│   │   └── collections.py            # Defines MongoDB collections
│   ├── cli/
│   │   └── main.py                   # CLI with .ccshare, MCP, LangChain support
│   ├── mcp/
│   │   ├── client.py                 # MCP client for connecting to external servers
│   │   ├── server.py                 # MCP server for ContextChain tools/data
│   │   └── tools.py                  # MCP-compatible tools (e.g., DB queries, APIs)
├── schemas/
│   ├── template.json                 # Schema template with MCP, LangChain fields
│   └── agent_bi.json                 # Example pipeline with MCP/LangChain tasks
├── config/
│   ├── default_config.yaml           # MongoDB, MCP, OpenAI API settings
│   └── team.ccshare                 # Atlas config for collaboration
├── examples/
│   ├── cashflow_agent_chain.json     # Sample with MCP, LangChain, OpenAI tasks
│   └── mcp_workflow.json             # Sample MCP-compatible workflow
├── tests/
│   ├── test_executor.py              # Tests executor with LangChain/MCP
│   ├── test_validator.py             # Tests schema validation
│   ├── test_version_manager.py       # Tests versioning
│   ├── test_mcp.py                   # Tests MCP client/server
│   └── test_langchain.py             # Tests LangChain chains/agents
├── .gitignore                        # Ignores .ccshare, env files
├── Dockerfile                        # For containerized deployment
├── LICENSE                           # MIT License
├── README.md                         # Updated with MCP, LangChain, OpenAI instructions
├── requirements.txt                  # Dependencies (add langchain, openai, mcp)
└── setup.py                          # For pip install



pymongo>=4.8.0
click>=8.1.7
pyyaml>=6.0.1
requests>=2.32.3
fastapi>=0.115.0
uvicorn>=0.30.6
langchain==0.3.1
langchain-openai==0.2.1
langchain-mcp-adapters>=0.1.0
openai>=1.7.0