"""LangChain prompt → Hugging Face Runnable → validated Pydantic output."""

from typing import TypeVar

from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, ValidationError

from mystery_graph import prompts
from mystery_graph.models import ChatBackend, ModelError, as_runnable
from mystery_graph.schemas import Answer, Evaluation, Hint, Mystery

T = TypeVar("T", bound=BaseModel)


class StructuredGenerationError(ModelError):
    """The model exhausted its bounded attempts to return valid structured output."""


class Generator:
    def __init__(self, backend: ChatBackend):
        self.model = as_runnable(backend)

    def _call(self, schema: type[T], system: str, user: str, values: dict) -> T:
        parser = PydanticOutputParser(pydantic_object=schema)
        prompt = ChatPromptTemplate.from_messages([("system", system), ("human", user)]).partial(
            format_instructions=parser.get_format_instructions()
        )
        chain = prompt | self.model | parser
        for attempt in range(2):
            try:
                return chain.invoke(
                    {
                        **values,
                        "repair": (
                            "The previous attempt failed schema validation. Produce a fresh valid JSON "
                            "object, with required fields, valid lengths and valid cross-references."
                            if attempt
                            else ""
                        ),
                    }
                )
            except (OutputParserException, ValidationError):
                continue
        raise StructuredGenerationError(
            "The model could not produce a valid response after two attempts. "
            "Try again or configure a stronger instruction-following model."
        )

    def story(self, **values) -> Mystery:
        return self._call(Mystery, prompts.STORY_SYSTEM, prompts.STORY_USER, values)

    def answer(self, **values) -> Answer:
        return self._call(Answer, prompts.ANSWER_SYSTEM, prompts.ANSWER_USER, values)

    def evaluate(self, **values) -> Evaluation:
        return self._call(Evaluation, prompts.EVALUATION_SYSTEM, prompts.EVALUATION_USER, values)

    def hint(self, **values) -> Hint:
        return self._call(Hint, prompts.HINT_SYSTEM, prompts.HINT_USER, values)
