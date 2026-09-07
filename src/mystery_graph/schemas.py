"""Validated boundaries between generated content, game logic, and the interface."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

ShortText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=600)]
Identifier = Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_]{0,39}$")]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Fact(StrictModel):
    id: Identifier
    text: ShortText


class Clue(StrictModel):
    id: Identifier
    fact_id: Identifier
    subtle: ShortText
    direct: ShortText


class Criterion(StrictModel):
    id: Identifier
    description: ShortText
    fact_ids: list[Identifier] = Field(min_length=1, max_length=5)


class Mystery(StrictModel):
    title: Annotated[str, StringConstraints(min_length=1, max_length=100)]
    scene: Annotated[str, StringConstraints(min_length=40, max_length=1800)]
    solution: Annotated[str, StringConstraints(min_length=20, max_length=1800)]
    facts: list[Fact] = Field(min_length=5, max_length=12)
    clues: list[Clue] = Field(min_length=3, max_length=6)
    criteria: list[Criterion] = Field(min_length=2, max_length=4)

    @model_validator(mode="after")
    def valid_references(self) -> "Mystery":
        for items in (self.facts, self.clues, self.criteria):
            if len({item.id for item in items}) != len(items):
                raise ValueError("IDs must be unique within each collection")
        fact_ids = {fact.id for fact in self.facts}
        if any(clue.fact_id not in fact_ids for clue in self.clues):
            raise ValueError("Clues must reference existing facts")
        if any(not set(c.fact_ids) <= fact_ids for c in self.criteria):
            raise ValueError("Criteria must reference existing facts")
        if self.solution.casefold() in self.scene.casefold():
            raise ValueError("The public scene must not contain the full solution")
        return self


class Answer(StrictModel):
    verdict: Literal["yes", "no", "unknown", "irrelevant"]
    evidence_ids: list[Identifier] = Field(default_factory=list, max_length=8)


class Match(StrictModel):
    criterion_id: Identifier
    quote: ShortText = Field(description="An exact quotation from the player's hypothesis")


class Evaluation(StrictModel):
    matches: list[Match] = Field(default_factory=list, max_length=4)


class Hint(StrictModel):
    text: ShortText


class GenerationRequest(StrictModel):
    theme: Annotated[str, StringConstraints(strip_whitespace=True, min_length=3, max_length=200)]
    difficulty: Literal["easy", "medium", "hard"] = "medium"


class TurnRequest(StrictModel):
    action: Literal["question", "hint", "hypothesis", "reveal"]
    text: Annotated[str, StringConstraints(strip_whitespace=True, max_length=1500)] = ""

    @model_validator(mode="after")
    def needs_text(self) -> "TurnRequest":
        if self.action in {"question", "hypothesis"} and not self.text:
            raise ValueError("Enter a question or hypothesis first.")
        return self
