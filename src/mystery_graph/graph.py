"""Explicit LangGraph transitions; each invocation returns one committed game state."""

import json
from copy import deepcopy
from typing import Literal, TypedDict
from uuid import uuid4

from langgraph.graph import END, START, StateGraph

from mystery_graph.generation import Generator
from mystery_graph.retrieval import Retrieval, serialize_documents
from mystery_graph.schemas import GenerationRequest, Mystery, TurnRequest


class GameState(TypedDict, total=False):
    action: str
    text: str
    theme: str
    difficulty: str
    seed: str
    references: str
    reference_ids: list[str]
    case: dict
    evidence: str
    evidence_ids: list[str]
    explored_fact_ids: list[str]
    matched_criteria: list[str]
    hint_keys: list[str]
    failed_attempts: int
    turns: int
    status: Literal["playing", "solved", "revealed"]
    reply: str
    history: list[dict[str, str]]


class GameError(ValueError):
    """Invalid game action safe to show in the interface."""


class MysteryGame:
    def __init__(self, generator: Generator, retrieval: Retrieval):
        self.generator = generator
        self.retrieval = retrieval
        builder = StateGraph(GameState)
        builder.add_node("retrieve_examples", self._retrieve_examples)
        builder.add_node("generate_story", self._generate_story)
        builder.add_node("retrieve_facts", self._retrieve_facts)
        builder.add_node("answer_question", self._answer_question)
        builder.add_node("generate_hint", self._generate_hint)
        builder.add_node("evaluate_hypothesis", self._evaluate_hypothesis)
        builder.add_node("reveal_solution", self._reveal_solution)
        builder.add_node("record_turn", self._record_turn)
        builder.add_conditional_edges(
            START,
            lambda state: state["action"],
            {
                "new": "retrieve_examples",
                "question": "retrieve_facts",
                "hint": "generate_hint",
                "hypothesis": "evaluate_hypothesis",
                "reveal": "reveal_solution",
            },
        )
        builder.add_edge("retrieve_examples", "generate_story")
        builder.add_edge("generate_story", END)
        builder.add_edge("retrieve_facts", "answer_question")
        for node in ("answer_question", "generate_hint", "evaluate_hypothesis", "reveal_solution"):
            builder.add_edge(node, "record_turn")
        builder.add_edge("record_turn", END)
        # Gradio holds each session's returned state on the server. There is no global game state.
        self.graph = builder.compile()

    def start(self, theme: str, difficulty: str = "medium") -> GameState:
        request = GenerationRequest(theme=theme, difficulty=difficulty)
        return self.graph.invoke(
            {
                "action": "new",
                "theme": request.theme,
                "difficulty": request.difficulty,
                "seed": uuid4().hex,
                "text": "",
            }
        )

    def turn(self, state: GameState | None, action: str, text: str = "") -> GameState:
        if not state or "case" not in state:
            raise GameError("Create a mystery first.")
        if state["status"] != "playing":
            raise GameError("This case is closed. Create a new mystery to continue.")
        if state["turns"] >= 40 and action != "reveal":
            raise GameError(
                "The 40-turn limit has been reached. Reveal the solution or start again."
            )
        request = TurnRequest(action=action, text=text)
        # A failed model call leaves the caller's current state intact, including counters/history.
        return self.graph.invoke(
            {
                **deepcopy(state),
                "action": request.action,
                "text": request.text,
            }
        )

    def _retrieve_examples(self, state: GameState) -> dict:
        docs = self.retrieval.references(f"{state['theme']} {state['difficulty']} lateral thinking")
        return {
            "references": serialize_documents(docs),
            "reference_ids": [d.metadata["reference_id"] for d in docs],
        }

    def _generate_story(self, state: GameState) -> dict:
        case = self.generator.story(
            theme=state["theme"],
            difficulty=state["difficulty"],
            seed=state["seed"],
            references=state["references"],
        )
        return {
            "case": case.model_dump(),
            "status": "playing",
            "turns": 0,
            "hint_keys": [],
            "failed_attempts": 0,
            "matched_criteria": [],
            "explored_fact_ids": [],
            "history": [],
            "evidence": "",
            "evidence_ids": [],
            "reply": "Ask yes/no questions, request a hint, or submit your explanation.",
        }

    def _retrieve_facts(self, state: GameState) -> dict:
        case = Mystery.model_validate(state["case"])
        # Include recent dialogue to resolve short follow-ups such as "Was it intentional?".
        context = " ".join(m["content"] for m in state["history"][-4:])
        docs = self.retrieval.evidence(case, f"{context}\n{state['text']}")
        return {
            "evidence": serialize_documents(docs),
            "evidence_ids": [d.metadata["fact_id"] for d in docs],
        }

    @staticmethod
    def _history(state: GameState) -> str:
        return json.dumps(state["history"][-12:], ensure_ascii=False)

    def _answer_question(self, state: GameState) -> dict:
        result = self.generator.answer(
            scene=state["case"]["scene"],
            evidence=state["evidence"],
            history=self._history(state),
            question=state["text"],
        )
        allowed = set(state["evidence_ids"])
        supported = bool(result.evidence_ids) and set(result.evidence_ids) <= allowed
        verdict = result.verdict
        if verdict in {"yes", "no"} and not supported:
            verdict = "unknown"
        replies = {
            "yes": "Yes.",
            "no": "No.",
            "unknown": "The available case facts do not establish that. Try a more specific question.",
            "irrelevant": "Please ask a yes/no question about the mystery.",
        }
        explored = set(state["explored_fact_ids"])
        if verdict in {"yes", "no"}:
            explored.update(result.evidence_ids)
        # No free-form model explanation can accidentally disclose the private solution here.
        return {"reply": replies[verdict], "explored_fact_ids": sorted(explored)}

    def _generate_hint(self, state: GameState) -> dict:
        case = Mystery.model_validate(state["case"])
        hints = set(state["hint_keys"])
        explored = set(state["explored_fact_ids"])
        clues = sorted(case.clues, key=lambda c: c.fact_id in explored)
        # Direct hints unlock earlier for easy games and after unsuccessful hypotheses.
        unlock_after = {"easy": 0, "medium": 1, "hard": 2}[state["difficulty"]]
        allow_direct = state["failed_attempts"] >= unlock_after or len(hints) >= len(clues)
        choices = [
            (clue, level)
            for level in ("subtle", "direct")
            for clue in clues
            if f"{clue.id}:{level}" not in hints and (level == "subtle" or allow_direct)
        ]
        # When stuck, prefer permitted direct clues before offering more subtle hints.
        if state["failed_attempts"] > unlock_after:
            choices.sort(key=lambda item: item[1] != "direct")
        if not choices:
            return {
                "reply": "You have seen every clue. Try an explanation, or reveal the solution."
            }
        clue, level = choices[0]
        generated = self.generator.hint(clue=getattr(clue, level), history=self._history(state))
        return {
            "reply": generated.text,
            "hint_keys": [*state["hint_keys"], f"{clue.id}:{level}"],
        }

    def _evaluate_hypothesis(self, state: GameState) -> dict:
        case = Mystery.model_validate(state["case"])
        result = self.generator.evaluate(
            solution=case.solution,
            facts=json.dumps(state["case"]["facts"]),
            criteria=json.dumps(state["case"]["criteria"]),
            hypothesis=state["text"],
        )
        allowed = {c.id for c in case.criteria}
        matches = {
            m.criterion_id
            for m in result.matches
            if m.criterion_id in allowed and m.quote.casefold() in state["text"].casefold()
        }
        solved = matches == allowed
        reply = (
            f"Solved!\n\n{case.solution}"
            if solved
            else f"You have connected {len(matches)} of {len(allowed)} key parts. "
            "Keep investigating how the events fit together."
        )
        return {
            "reply": reply,
            "status": "solved" if solved else "playing",
            "matched_criteria": sorted(matches),
            "failed_attempts": state["failed_attempts"] + (0 if solved else 1),
        }

    def _reveal_solution(self, state: GameState) -> dict:
        return {"reply": state["case"]["solution"], "status": "revealed"}

    def _record_turn(self, state: GameState) -> dict:
        labels = {"hint": "Give me a hint.", "reveal": "Reveal the solution."}
        user_text = labels.get(state["action"], state["text"])
        if state["action"] == "hypothesis":
            user_text = f"My hypothesis: {user_text}"
        return {
            "turns": state["turns"] + 1,
            "history": [
                *state["history"],
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": state["reply"]},
            ],
            "references": "",
            "evidence": "",
            "evidence_ids": [],
        }


def public_view(state: GameState | None) -> dict:
    """Only this allowlisted projection is sent to visible Gradio components."""
    if not state:
        return {
            "title": "Your next mystery",
            "scene": "Choose a setting to open a case.",
            "history": [],
            "status": "No active case",
        }
    return {
        "title": state["case"]["title"],
        "scene": state["case"]["scene"],
        "history": deepcopy(state["history"]),
        "status": f"{state['status'].capitalize()} · {state['turns']}/40 turns · "
        f"{len(state['hint_keys'])} hints · {state['failed_attempts']} unsuccessful hypotheses",
    }
