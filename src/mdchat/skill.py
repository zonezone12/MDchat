"""
Skill abstraction layer for MDChat.

Defines the Skill ABC, Parameter schema, SkillResult, and the @md_skill
decorator for wrapping functions as skills.
"""

from __future__ import annotations

import functools
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    TYPE_CHECKING,
)

if TYPE_CHECKING:
    from .context import AnalysisContext


class ParamType(Enum):
    """JSON-Schema-compatible parameter types with MD-domain extensions."""

    STRING = "string"
    INTEGER = "integer"
    FLOAT = "number"
    BOOLEAN = "boolean"
    ARRAY = "array"
    # Domain-specific (serialized as string with semantic meaning)
    ATOM_SELECTION = "string"
    FILE_PATH = "string"


@dataclass
class Parameter:
    """Typed parameter descriptor for a Skill."""

    name: str
    param_type: ParamType
    description: str
    required: bool = True
    default: Any = None
    enum_values: Optional[List[str]] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    items_type: Optional[ParamType] = None  # for ARRAY params

    def to_json_schema(self) -> dict:
        """Convert to a JSON Schema property definition."""
        schema: Dict[str, Any] = {
            "type": self.param_type.value,
            "description": self.description,
        }
        if self.enum_values:
            schema["enum"] = self.enum_values
        if self.min_value is not None:
            schema["minimum"] = self.min_value
        if self.max_value is not None:
            schema["maximum"] = self.max_value
        if self.default is not None:
            schema["default"] = self.default
        if self.param_type == ParamType.ARRAY and self.items_type:
            schema["items"] = {"type": self.items_type.value}
        return schema


@dataclass
class SkillResult:
    """Structured result returned by a Skill execution."""

    success: bool
    data: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, str] = field(default_factory=dict)  # name -> file path
    summary: str = ""
    error: Optional[str] = None

    def to_tool_result(self) -> str:
        """Format as a concise string for the LLM tool_result message."""
        parts = []
        if self.summary:
            parts.append(self.summary)
        if self.artifacts:
            parts.append("Generated files: " + ", ".join(
                f"{k}: {v}" for k, v in self.artifacts.items()
            ))
        if self.error:
            parts.append(f"Error: {self.error}")
        if not parts:
            parts.append("Done (no additional output).")
        return "\n".join(parts)


class Skill(ABC):
    """
    Abstract base class for an MDChat skill.

    Each skill is a self-describing, LLM-callable analytical unit that wraps
    one or more operations from the MD_analysis library.
    """

    name: str = ""
    description: str = ""
    category: str = "general"
    parameters: List[Parameter] = []
    requires: List[str] = []
    produces: List[str] = []

    def validate(self, context: AnalysisContext) -> Tuple[bool, str]:
        """Check whether this skill can execute given the current context.

        Returns (ok, reason). Default implementation checks that every key
        listed in *requires* is present in the context.
        """
        missing = [r for r in self.requires if not context.has(r)]
        if missing:
            return False, f"Missing prerequisites: {', '.join(missing)}"
        return True, ""

    @abstractmethod
    def execute(self, context: AnalysisContext, **params: Any) -> SkillResult:
        """Run the analysis and return structured results."""
        ...

    def to_tool_schema(self) -> dict:
        """Generate an Anthropic-compatible tool definition."""
        properties: Dict[str, Any] = {}
        required: List[str] = []

        for p in self.parameters:
            properties[p.name] = p.to_json_schema()
            if p.required:
                required.append(p.name)

        input_schema: Dict[str, Any] = {
            "type": "object",
            "properties": properties,
        }
        if required:
            input_schema["required"] = required

        return {
            "name": self.name,
            "description": self.description,
            "input_schema": input_schema,
        }


# ---------------------------------------------------------------------------
# @md_skill decorator – convenience for wrapping a plain function as a Skill
# ---------------------------------------------------------------------------

def md_skill(
    name: str,
    description: str,
    parameters: List[Parameter] | None = None,
    requires: List[str] | None = None,
    produces: List[str] | None = None,
    category: str = "general",
) -> Callable:
    """Decorator that turns a plain function into a Skill instance.

    The decorated function must have the signature::

        def my_skill(context: AnalysisContext, **params) -> SkillResult:
            ...

    Example::

        @md_skill(
            name="compute_rg",
            description="Compute radius of gyration over trajectory",
            parameters=[Parameter("selection", ParamType.STRING, "Atom selection")],
            requires=["universe"],
            produces=["rg_array"],
        )
        def compute_rg(context, selection="protein"):
            ...
    """

    def decorator(fn: Callable) -> "_FunctionSkill":
        skill = _FunctionSkill(
            _name=name,
            _description=description,
            _parameters=parameters or [],
            _requires=requires or [],
            _produces=produces or [],
            _category=category,
            _fn=fn,
        )
        functools.update_wrapper(skill, fn)
        return skill

    return decorator


class _FunctionSkill(Skill):
    """Skill implementation produced by @md_skill."""

    def __init__(
        self,
        _name: str,
        _description: str,
        _parameters: List[Parameter],
        _requires: List[str],
        _produces: List[str],
        _category: str,
        _fn: Callable,
    ) -> None:
        self.name = _name
        self.description = _description
        self.parameters = _parameters
        self.requires = _requires
        self.produces = _produces
        self.category = _category
        self._fn = _fn

    def execute(self, context: AnalysisContext, **params: Any) -> SkillResult:
        try:
            return self._fn(context, **params)
        except Exception as exc:
            return SkillResult(
                success=False,
                error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
                summary=f"Skill '{self.name}' failed: {exc}",
            )
