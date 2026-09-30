"""不带RemainingSteps"""
# from typing_extensions import TypedDict
# from langgraph.graph import StateGraph
# from langgraph.graph import START, END

# class State(TypedDict):
#     value: str
#     action_result: str


# def router(state: State):
#     if state["value"] == "end":
#         return END
#     else:
#         return "action"

# def decision_node(state):
#     return {"value": "keep going!"}


# def action_node(state: State):
#     # Do your action here ...
#     return {"action_result": "what a great result!"}


# workflow = StateGraph(State)
# workflow.add_node("decision", decision_node)
# workflow.add_node("action", action_node)
# workflow.add_edge(START, "decision")
# workflow.add_conditional_edges("decision", router, ["action", END])
# workflow.add_edge("action", "decision")
# app = workflow.compile()

# app.get_graph().draw_mermaid_png(output_file_path='../imgs/示例5.3.png')

# from langgraph.errors import GraphRecursionError

# try:
#     app.invoke({"value": "hi!"})
# except GraphRecursionError:
#     print("Recursion Error")

"""带RemainingSteps"""

from langgraph.constants import START, END
from typing_extensions import TypedDict
from langgraph.graph import StateGraph
from typing import Annotated
from langgraph.managed.is_last_step import RemainingSteps

class State(TypedDict):
    value: str
    action_result: str
    remaining_steps: RemainingSteps

def router(state: State):
    if state["remaining_steps"] <= 2:
        return END
    if state["value"] == "end":
        return END
    else:
        return "action"

def decision_node(state):
    return {"value": "keep going!"}

def action_node(state: State):
    return {"action_result": "what a great result!"}


workflow = StateGraph(State)
workflow.add_node("decision", decision_node)
workflow.add_node("action", action_node)
workflow.add_edge(START, "decision")
workflow.add_conditional_edges("decision", router, ["action", END])
workflow.add_edge("action", "decision")
app = workflow.compile()

app.get_graph().draw_mermaid_png(output_file_path='../../../../../assets/示例5.3.png')

from langgraph.errors import GraphRecursionError
try:
    print(app.invoke({"value": "hi!"}))
except GraphRecursionError:
    print("Recursion Error")