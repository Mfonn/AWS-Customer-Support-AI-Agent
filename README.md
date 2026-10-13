# AI Customer Support Agent on Amazon Bedrock AgentCore

A customer support agent for an e-commerce platform, built with **Amazon Bedrock AgentCore** and the **Strands SDK**. It tracks orders, processes refunds, answers product and policy questions from a knowledge base, remembers customers across sessions, calculates loyalty discounts with exact arithmetic, and can read live web pages.

Built as the capstone project of Udacity's *Future AWS Agent Engineer* programme. The starter skeleton and the two Lambda functions come from the course; the agent implementation in `main.py` is my own.

## What it does

- **Orders and refunds**: calls Lambda-backed tools (order tracking, refund initiation, refund status, return labels) through AgentCore Gateway using the Model Context Protocol.
- **Grounded answers**: retrieves product, warranty and loyalty-programme information from a Bedrock Knowledge Base instead of answering from model memory.
- **Cross-session memory**: recalls a customer's name and preferences in a brand-new session.
- **Exact discount maths**: runs loyalty-discount calculations as Python inside a sandbox.
- **Live web access**: opens an isolated managed browser session for pages the customer asks about.

## Architecture

```
Customer request
      │
      ▼
AgentCore Runtime ── Strands Agent (Amazon Nova 2 Lite)
      │
      ├─► AgentCore Gateway (MCP) ─► API Gateway ─► Lambda: order-tracker
      │                           └─► Lambda: refund-processor
      ├─► search_knowledge_base ───► Bedrock Knowledge Base (product catalog in S3)
      ├─► MemoryHook ──────────────► AgentCore Memory (semantic + user-preference)
      ├─► calculate_loyalty_discount ─► AgentCore Code Interpreter (sandbox)
      └─► AgentCore Browser ───────► live web pages
```

| Component | Role | Where in the code |
|---|---|---|
| AgentCore Runtime | Hosts the agent; one async entrypoint per request | `@app.entrypoint` in `invoke()` |
| Gateway + MCP | Exposes Lambdas as tools the agent discovers at runtime | `MCPClient` inside `invoke()` |
| Knowledge Base (RAG) | Grounds policy and product answers | `search_knowledge_base()` |
| AgentCore Memory | Retrieves context before a reply, saves the turn after | `MemoryHook`, `get_namespaces()` |
| Code Interpreter | Deterministic loyalty-discount calculation | `calculate_loyalty_discount()` |
| Browser tool | Reads live web pages | `AgentCoreBrowser` in `invoke()` |

## Repository structure

```
main.py                 Agent: tools, memory hook, entrypoint
pyproject.toml          Dependencies (Python 3.14+)
product_catalog.txt     Knowledge Base source document
lambda/                 order_tracker.py, refund_processor.py, lambda_schema
```

## Setup and deployment

**Prerequisites:** Python 3.14 (via [uv](https://docs.astral.sh/uv/)), AWS CLI v2, an AWS account with Bedrock model access to Amazon Nova, region `us-east-1`.

1. Install dependencies:
   ```bash
   uv python install 3.14
   uv sync --python 3.14
   ```
2. Create the AWS resources (console): the two Lambda functions, a REST API in API Gateway with the three GET operations (`get_order`, `get_customer_orders`, `get_customer`), an AgentCore Gateway with an API Gateway target and a Lambda target (target names must be hyphenated), a Bedrock Knowledge Base synced from `product_catalog.txt`, and an AgentCore Memory resource with semantic and user-preference strategies.
3. Set the four values near the top of `main.py`: `GATEWAY_URL`, `KB_ID`, `REGION`, `MEMORY_ID`.
4. Deploy:
   ```bash
   uv run agentcore configure --entrypoint main.py --name CustomerSupportAgent
   uv run agentcore deploy
   ```
5. The agent's runtime role needs permission to call `bedrock:Retrieve` on the Knowledge Base, the AgentCore browser and code-interpreter actions, and the memory actions (`CreateEvent`, `RetrieveMemoryRecords`, `GetMemory`).

Invoke the deployed agent:
```bash
uv run agentcore invoke '{"prompt": "Can you track order ORD-001?", "customer_id": "CUST-123", "session_id": "t1"}'
```

## Test scenarios

| # | Scenario | Example prompt |
|---|---|---|
| 1 | Order tracking | "Can you track order ORD-001?" |
| 2 | Refund processing | "I want to return my Kindle Paperwhite (ORD-002). Please initiate a refund." |
| 3 | Knowledge Base (RAG) | "What are the benefits of the Platinum loyalty tier?" |
| 4 | Long-term memory | Session A: "Hi, I am Jane. I prefer concise responses." Wait about 30 seconds, then in a new session: "Do you remember my name and communication preference?" |
| 5 | Loyalty discount | "I am a Gold member with 4250 points. Calculate my discount on a $150 standard order." |
| 6 | Browser tool | "Go to https://www.udacity.com and tell me the page title." |

## Reflection

**Design decision: exact arithmetic in the Code Interpreter.** For `calculate_loyalty_discount` I kept the language model out of the maths on purpose. A model asked to redeem points and then apply a tier discount can return a plausible but wrong total, and in a billing context that is a real error. The tool instead builds a self-contained Python script, runs it in an isolated AgentCore Code Interpreter session with `clearContext=True` so no state leaks between customers, and returns the JSON the script prints. Because customer input ends up inside generated code, I validate the tier, category and numbers first and inject them as `repr()` literals rather than raw text. If the sandbox is unavailable, a fallback applies only the tier discount and says so.

**Challenge: a permissions error hidden behind a friendly message.** After deployment, the Knowledge Base question returned "the knowledge base is temporarily unavailable". That was my own fallback text, so it concealed the real exception. I had already fixed one permission problem on the Gateway's execution role, which could not invoke the refund Lambda, so I first looked in the wrong place. The agent's runtime role is a separate identity and also needed `bedrock:Retrieve`. I attached a customer-managed policy to the runtime's role and the query then succeeded. I also learned that IAM rejects action names that do not exist: `RetrieveMemories` had to be `RetrieveMemoryRecords`.

**Production consideration: security.** Several choices here are sandbox-only. The Gateway uses the `NONE` authorizer, and I used a wildcard resource on the browser permission to unblock testing. In production I would put the Gateway behind OAuth or IAM authorisation, scope every IAM statement to specific ARNs, and restrict the browser tool to an allowlist of domains, because a page the agent reads is untrusted input that could carry a prompt injection. I would also add CloudWatch alarms on tool-call errors, so a failure like the Knowledge Base one surfaces within minutes instead of when a customer complains.

## Notes

- This was built in a temporary training sandbox. Do not reuse its permissive settings (no-auth Gateway, wildcard resources) for real customer data.
- Account IDs, resource IDs and credentials are deliberately not committed.
