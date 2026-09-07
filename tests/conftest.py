"""Synthetic fixtures: no inference, model downloads, API keys, or recorded live outputs."""

import socket
from copy import deepcopy

import pytest
from langchain_core.documents import Document

from mystery_graph.graph import MysteryGame
from mystery_graph.schemas import Answer, Evaluation, Hint, Mystery


@pytest.fixture(autouse=True)
def deny_network(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("Tests must not use the network")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)


@pytest.fixture
def case():
    return Mystery(
        title="The Sealed Box",
        scene="An archivist waits beside a closed box before examining a valuable letter. Why?",
        solution="The box is cold from storage. Warming it while sealed prevents condensation on the letter.",
        facts=[
            {"id": "f_cold", "text": "The box has just left cold storage."},
            {"id": "f_sealed", "text": "The lid is sealed while the box warms."},
            {"id": "f_air", "text": "Warm room air contains moisture."},
            {"id": "f_condensation", "text": "Moist air condenses on cold surfaces."},
            {"id": "f_damage", "text": "The letter can be damaged by water."},
        ],
        clues=[
            {
                "id": "c_temperature",
                "fact_id": "f_cold",
                "subtle": "Consider its previous location.",
                "direct": "The box arrived from somewhere cold.",
            },
            {
                "id": "c_air",
                "fact_id": "f_air",
                "subtle": "Consider what is in the room's air.",
                "direct": "The surrounding air contains moisture.",
            },
            {
                "id": "c_letter",
                "fact_id": "f_damage",
                "subtle": "Think about the letter's fragility.",
                "direct": "Water could damage the paper.",
            },
        ],
        criteria=[
            {"id": "cold", "description": "Identify the cold box.", "fact_ids": ["f_cold"]},
            {
                "id": "condensation",
                "description": "Explain the condensation risk.",
                "fact_ids": ["f_condensation", "f_damage"],
            },
        ],
    )


class StubGenerator:
    def __init__(self, case):
        self.case = case
        self.next_answer = Answer(verdict="yes", evidence_ids=["f_cold"])
        self.next_evaluation = Evaluation(matches=[])
        self.fail = False
        self.calls = []

    def story(self, **kwargs):
        self.calls.append(("story", kwargs))
        return deepcopy(self.case)

    def answer(self, **kwargs):
        self.calls.append(("answer", kwargs))
        if self.fail:
            raise RuntimeError("Synthetic provider failure")
        return self.next_answer

    def evaluate(self, **kwargs):
        self.calls.append(("evaluate", kwargs))
        return self.next_evaluation

    def hint(self, **kwargs):
        self.calls.append(("hint", kwargs))
        return Hint(text=kwargs["clue"])


class StubRetrieval:
    def references(self, query):
        return [Document(page_content="Synthetic reference", metadata={"reference_id": "ref_test"})]

    def evidence(self, case, query):
        return [Document(page_content=case.facts[0].text, metadata={"fact_id": case.facts[0].id})]


@pytest.fixture
def game(case):
    generator = StubGenerator(case)
    return MysteryGame(generator, StubRetrieval()), generator
