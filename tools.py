
import ast
import math
import operator
from contextvars import ContextVar
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

from langchain_core.tools import tool
from langchain_tavily import TavilySearch


# Current conversation
_current_thread_id = ContextVar(
    "wizora_thread_id",
    default="default"
)


def set_current_thread_id(thread_id: str):
    _current_thread_id.set(thread_id or "default")


def get_current_thread_id():
    return _current_thread_id.get()


# Web search
web_search = TavilySearch(
    max_results=3,
    topic="general",
    search_depth="basic"
)


# Calculator
_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow
}

_UNARY_OPERATORS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg
}

_ALLOWED_FUNCTIONS = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum
}

_ALLOWED_MATH_FUNCTIONS = {
    "sqrt": math.sqrt,
    "ceil": math.ceil,
    "floor": math.floor,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "radians": math.radians,
    "degrees": math.degrees
}

_ALLOWED_MATH_CONSTANTS = {
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau
}


def _evaluate_expression(node):

    if isinstance(node, ast.Expression):
        return _evaluate_expression(node.body)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value

        raise ValueError("Only numeric values are allowed.")

    if isinstance(node, ast.Name):
        if node.id in _ALLOWED_MATH_CONSTANTS:
            return _ALLOWED_MATH_CONSTANTS[node.id]

        raise ValueError("Unknown constant.")

    if isinstance(node, ast.BinOp):
        operation = _BINARY_OPERATORS.get(type(node.op))

        if operation is None:
            raise ValueError("Unsupported mathematical operation.")

        left = _evaluate_expression(node.left)
        right = _evaluate_expression(node.right)

        if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
            raise ValueError("Only numbers can be used in calculations.")

        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("Exponent is too large.")

        result = operation(left, right)

        if isinstance(result, complex):
            raise ValueError("Complex results are not supported.")

        if isinstance(result, int) and result.bit_length() > 4096:
            raise ValueError("Result is too large.")

        if isinstance(result, float) and not math.isfinite(result):
            raise ValueError("Result is not finite.")

        return result

    if isinstance(node, ast.UnaryOp):
        operation = _UNARY_OPERATORS.get(type(node.op))

        if operation is None:
            raise ValueError("Unsupported unary operation.")

        return operation(_evaluate_expression(node.operand))

    if isinstance(node, (ast.List, ast.Tuple)):
        if len(node.elts) > 100:
            raise ValueError("Too many values.")

        return [
            _evaluate_expression(item)
            for item in node.elts
        ]

    if isinstance(node, ast.Call):

        if isinstance(node.func, ast.Name):
            function = _ALLOWED_FUNCTIONS.get(node.func.id)

        elif (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "math"
        ):
            function = _ALLOWED_MATH_FUNCTIONS.get(node.func.attr)

        else:
            function = None

        if function is None:
            raise ValueError("Function is not allowed.")

        if node.keywords:
            raise ValueError("Keyword arguments are not supported.")

        arguments = [
            _evaluate_expression(arg)
            for arg in node.args
        ]

        return function(*arguments)

    if isinstance(node, ast.Attribute):
        if (
            isinstance(node.value, ast.Name)
            and node.value.id == "math"
            and node.attr in _ALLOWED_MATH_CONSTANTS
        ):
            return _ALLOWED_MATH_CONSTANTS[node.attr]

    raise ValueError("Invalid mathematical expression.")


@tool
def calculator(expression: str) -> str:
    """
    Calculate mathematical expressions.

    Examples:
    2 + 2
    math.sqrt(16)
    10 * 5
    100 / 4
    """

    try:
        if not expression or len(expression) > 200:
            raise ValueError("Invalid expression length.")

        parsed_expression = ast.parse(
            expression,
            mode="eval"
        )

        if sum(1 for _ in ast.walk(parsed_expression)) > 50:
            raise ValueError("Expression is too complex.")

        result = _evaluate_expression(parsed_expression)

        return str(result)

    except Exception as e:
        return f"Calculation error: {str(e)}"


# Search uploaded documents
@tool
def search_uploaded_documents(query: str) -> str:
    """
    Search uploaded PDFs, DOCX, TXT, and other supported documents
    for relevant information.
    """

    from rag import retrieve_from_rag

    return retrieve_from_rag(
        query=query,
        thread_id=get_current_thread_id()
    )


# Save user memory
@tool
def remember_this(memory: str) -> str:
    """
    Save an important user preference or fact
    into long-term memory.
    """

    from database import init_db, save_memory

    init_db()

    return save_memory(
        thread_id=get_current_thread_id(),
        memory=memory
    )


# Retrieve saved memory
@tool
def recall_memory(query: str) -> str:
    """
    Retrieve previously saved information
    about the current user conversation.
    """

    from database import init_db, search_memory

    init_db()

    return search_memory(
        thread_id=get_current_thread_id(),
        query=query
    )


# Available tools
tools = [
    calculator,
    search_uploaded_documents,
    remember_this,
    recall_memory,
    web_search
]
