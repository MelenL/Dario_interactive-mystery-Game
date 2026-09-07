"""Local-first Gradio interface; models initialize only on the first requested game."""

import logging
from threading import Lock

import gradio as gr
from pydantic import ValidationError

from mystery_graph.config import ConfigurationError, Settings
from mystery_graph.generation import Generator
from mystery_graph.graph import GameError, MysteryGame, public_view
from mystery_graph.models import ModelError, create_backend
from mystery_graph.retrieval import Retrieval

LOGGER = logging.getLogger(__name__)
CSS = """
.gradio-container { max-width: 1060px !important; margin: auto; }
#case-scene textarea { font-size: 1.05rem; line-height: 1.6; }
"""


class LazyGame:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._game = None
        self._lock = Lock()

    def get(self) -> MysteryGame:
        with self._lock:
            if self._game is None:
                self.settings.require_generation_credentials()
                backend = create_backend(self.settings)
                retrieval = Retrieval.from_settings(self.settings)
                self._game = MysteryGame(Generator(backend), retrieval)
            return self._game


def build_app(settings: Settings) -> gr.Blocks:
    lazy_game = LazyGame(settings)

    def present(state):
        view = public_view(state)
        return state, view["title"], view["scene"], view["history"], view["status"], ""

    def handle_error(exc: Exception):
        if isinstance(exc, ValidationError):
            # Do not echo the validation input: it can include private model output.
            raise gr.Error(
                "Check the input: setting 3–200 characters; question/hypothesis 1–1500."
            ) from None
        if isinstance(exc, (ConfigurationError, ModelError, GameError)):
            raise gr.Error(str(exc)) from None
        # Avoid logging prompts, answers, tokens, model responses or traceback payloads.
        LOGGER.error("Game operation failed (%s).", type(exc).__name__)
        raise gr.Error(
            "The operation could not finish. Your current case has been preserved."
        ) from None

    def new_game(theme, difficulty):
        try:
            return present(lazy_game.get().start(theme, difficulty))
        except Exception as exc:
            handle_error(exc)

    def take_turn(action, state, text=""):
        try:
            if not state:
                raise GameError("Create a mystery first.")
            return present(lazy_game.get().turn(state, action, text))
        except Exception as exc:
            handle_error(exc)

    def ask(state, text):
        return take_turn("question", state, text)

    def hypothesize(state, text):
        return take_turn("hypothesis", state, text)

    def hint(state):
        return take_turn("hint", state)

    def reveal(state):
        return take_turn("reveal", state)

    with gr.Blocks(title="Mystery Graph", analytics_enabled=False) as app:
        gr.Markdown(
            "# Mystery Graph\nA strange scene. A hidden explanation. Your questions connect them."
        )
        state = gr.State(value=None, time_to_live=3600)
        with gr.Row():
            theme = gr.Textbox(
                label="Setting", value="An observatory on a foggy island", max_lines=2
            )
            difficulty = gr.Radio(["easy", "medium", "hard"], value="medium", label="Difficulty")
        new = gr.Button("Create a mystery", variant="primary")
        title = gr.Textbox(value="Your next mystery", label="Case", interactive=False)
        scene = gr.Textbox(
            value="Choose a setting to open a case.",
            label="The scene",
            lines=5,
            interactive=False,
            elem_id="case-scene",
        )
        status = gr.Textbox(value="No active case", label="Progress", interactive=False)
        chat = gr.Chatbot(
            label="Investigation",
            height=340,
            render_markdown=False,
            allow_file_downloads=False,
        )
        text = gr.Textbox(
            label="Your question or explanation", placeholder="Was the timing important?", lines=2
        )
        with gr.Row():
            question = gr.Button("Ask a question", variant="primary")
            hypothesis = gr.Button("Submit hypothesis")
            clue = gr.Button("Request a hint")
        with gr.Accordion("Ready to see the answer?", open=False):
            gr.Markdown("Revealing the explanation closes this case.")
            give_up = gr.Button("Reveal solution")
        gr.Markdown(
            "AI-generated fiction. Answers may be imperfect. A new case resets the investigation."
        )
        outputs = [state, title, scene, chat, status, text]
        event_args = dict(
            outputs=outputs, concurrency_id="game", concurrency_limit=1, api_visibility="private"
        )
        new.click(new_game, inputs=[theme, difficulty], **event_args)
        question.click(ask, inputs=[state, text], **event_args)
        text.submit(ask, inputs=[state, text], **event_args)
        hypothesis.click(hypothesize, inputs=[state, text], **event_args)
        clue.click(hint, inputs=[state], **event_args)
        give_up.click(reveal, inputs=[state], **event_args)
    return app


def main() -> None:
    settings = Settings.from_env()
    app = build_app(settings)
    app.queue(max_size=20, default_concurrency_limit=1).launch(
        server_name=settings.host,
        server_port=settings.port,
        share=False,
        auth=(settings.username, settings.password) if settings.username else None,
        show_error=False,
        theme=gr.themes.Soft(),
        css=CSS,
        state_session_capacity=100,
    )


if __name__ == "__main__":
    main()
