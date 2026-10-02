from typing import TypedDict
from langgraph.graph import StateGraph, END
from laya.integrations.langchain import LayaRouter

class AgentState(TypedDict):
    input: str
    response: str

# Define router with confidence threshold fallback
router = LayaRouter(
    criteria={
        "billing_agent": "invoices, payment methods, duplicate charges, refunds",
        "tech_support": "system errors, bugs, API downtime, stack traces",
        "sales_agent": "pricing plans, new contracts, demo requests",
    },
    instructions="Which specialist agent should answer this user query?",
    confidence_threshold=0.80,   # If answer_confidence < 0.80, route to the fallback
    fallback="human_agent",
    state_key="input",
)

workflow = StateGraph(AgentState)

# Add specialist nodes
workflow.add_node("billing_agent", lambda state: {"response": "Handling billing..."})
workflow.add_node("tech_support", lambda state: {"response": "Handling tech support..."})
workflow.add_node("sales_agent", lambda state: {"response": "Handling sales..."})
workflow.add_node("human_agent", lambda state: {"response": "Escalated to human support."})

# Add conditional edge using LayaRouter
workflow.set_conditional_entry_point(
    lambda state: router(state),
    {
        "billing_agent": "billing_agent",
        "tech_support": "tech_support",
        "sales_agent": "sales_agent",
        "human_agent": "human_agent",
    }
)

app = workflow.compile()
result = app.invoke({"input": "I was billed twice for last month's subscription."})
print(result["response"])  # -> "Handling billing..."

route_decision = router.last_decision["answers"]["route"]
print("Answers are:", router.last_decision["answers"])
print("Predicted route:", route_decision["choice"])
print(
    "Route confidence:",
    route_decision.get("answer_confidence", route_decision.get("confidence")),
)
print("Route probabilities:", route_decision.get("probabilities"))


#the probabilities of each route choice is checked against the confidence threshold to determine if the fallback should be used