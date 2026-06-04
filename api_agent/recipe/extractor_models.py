"""Structured recipe extractor output models."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class RecipeToolArgOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: str
    description: str | None = None


class RecipeInputValueOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: str
    transform: str | None = None


class RecipeStepInputOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    mode: str
    from_: str | None = Field(default=None, alias="from")
    bind: dict[str, str] | None = None
    with_: dict[str, RecipeInputValueOutput] = Field(default_factory=dict, alias="with")


class RecipeCallOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query_template: str | None = None
    method: str | None = None
    path: str | None = None
    path_params: dict[str, Any] | None = None
    query_params: dict[str, Any] | None = None
    body: dict[str, Any] | list[Any] | str | int | float | bool | None = None


class RecipeStepOutputSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    attach_binding: list[str] | None = None


class RecipeStepOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    kind: str
    input: RecipeStepInputOutput
    output: RecipeStepOutputSpec
    query_template: str | None = None
    call: RecipeCallOutput | None = None


class RecipePublicContractOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool_name: str
    description: str
    tool_args: dict[str, RecipeToolArgOutput]


class RecipeExecutionPlanOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    steps: list[RecipeStepOutput]


class RecipeValidationFixtureOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool_args: dict[str, Any] = Field(default_factory=dict)
    result: Any | None = None


class ExtractedRecipeOutput(BaseModel):
    """Extractor output is only a candidate recipe, never a save decision."""

    model_config = ConfigDict(extra="forbid")

    public_contract: RecipePublicContractOutput
    execution_plan: RecipeExecutionPlanOutput
    validation_fixture: RecipeValidationFixtureOutput = Field(
        default_factory=RecipeValidationFixtureOutput
    )
