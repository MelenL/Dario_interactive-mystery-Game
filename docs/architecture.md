# Architecture notes

## State and graph boundaries

`MysteryGame` owns one compiled `StateGraph`. The graph is reusable across sessions because
it does not store a mutable current game on the instance. Each `start()` returns a fresh
`GameState`; each `turn()` invokes the graph with a deep copy of the existing state.

`gr.State` holds the returned dictionary on the server for that browser session. Only
`public_view()` projects data into the interface. The private solution, rubric, canonical
facts, retrieved references, and retrieval payloads are not visible components. The explanation
enters the dialogue only after a solved hypothesis or the explicit reveal action.

There is deliberately no database or checkpointer in the default configuration. A failed
invocation does not replace the session's previous state, so retries do not increment counters
or append duplicate messages. This provides atomic state replacement at the application boundary,
not distributed transactions or exactly-once inference: a failed request may still be billed.

LangGraph routes explicit user actions rather than using an additional model call to guess
whether a message is a question or hypothesis. The interface makes the distinction visible.

## Structured generation

Each task is a LangChain chain:

```text
ChatPromptTemplate → RunnableLambda(Hugging Face backend) → PydanticOutputParser
```

The same adapter supports hosted `InferenceClient.chat_completion` or a local Transformers
text-generation pipeline. JSON schemas travel in the prompt; provider-specific JSON mode is
not required. Invalid structured output triggers at most one fresh regeneration. Authentication,
timeout, or quota failures do not trigger a schema retry. Raw provider error bodies are not
displayed because they can contain private prompts or credentials.

The story schema checks field lengths, collection sizes, unique IDs, and references from clues
and criteria to canonical facts. It also rejects a scene containing the complete solution
verbatim. These are structural checks, not a semantic consistency proof.

## Retrieval design

Eight small original reference mysteries provide patterns such as changed viewpoints,
unusual object functions, and timing constraints. Their role is contextual inspiration rather
than training data or a benchmark. A new scene need not reproduce a retrieved story.

Each generated case contains 5–12 atomic facts. A lazily built vector index embeds those facts
and retrieves the nearest ones for the current question plus the last two exchanges. The
cache holds at most 32 case indexes and includes the complete serialized case in its key.
Cache eviction causes re-embedding on the next question but does not delete session state.
Cached indexes remain in process memory until eviction or restart even if a browser closes.

Semantic relevance is not sufficient evidence by itself. The model returns a verdict and fact
IDs; a yes/no verdict without valid IDs from the retrieved context becomes `unknown`. A
reference-ID check cannot prove that a model correctly interpreted the cited fact.

## Adaptive hints

The generated case includes ordered clue pairs: subtle and direct. The selection policy uses:

- previously served clue/level pairs, to avoid repeating the same prompt;
- facts explored in yes/no answers, to prioritize less-explored lines of investigation;
- difficulty and unsuccessful hypothesis counts, to unlock stronger hints;
- exhaustion of subtle clues, so difficult games cannot get stuck without a direct hint.

For easy, medium, and hard cases, direct hints become available after 0, 1, or 2 unsuccessful
hypotheses respectively. After more unsuccessful attempts, direct hints take priority. Otherwise
subtle hints are preferred. The model rephrases only the selected permitted clue using recent
dialogue; it is not given the hidden explanation or rubric on this path.

`explored_fact_ids` records which facts supported an answer, not a proven estimate of player
knowledge. Different clues can overlap semantically even though their IDs are unique.

## Hypothesis evaluation

The evaluator sees the full fixed explanation and 2–4 generated rubric criteria. It returns
criterion IDs with literal quotes from the current hypothesis. The application rejects unknown
IDs and quotations absent from the input, deduplicates matches, and awards victory only if all
criteria match in that single attempt. It never merges partial matches from different attempts.

Unsuccessful feedback gives only the number of matched criteria. The evaluator's free-form
reasoning is never exposed. Quotation checks reduce fabricated evidence but cannot guarantee
correct grading. Negations, contradictory guesses, and adversarial inputs require live testing.

## Resource and disclosure limits

- Theme length: 3–200 characters; question/hypothesis length: 1–1500 characters.
- Investigation: 40 turns plus an optional final reveal.
- Prompt dialogue: last 12 messages; displayed history: the bounded complete game.
- Queue: 20 pending actions, one concurrent game operation globally.
- Hosted timeout: configurable, default 90 seconds per request. Local inference has no hard
  interruption timeout and can block the queue on slow hardware.
- Session state: up to 100 Gradio sessions; one-hour state TTL.
- Textboxes and non-Markdown chat render generated text without fetching generated images or links.
- Generation and embedding model loading is deferred; no inference happens on module import.

Prompt instructions are not a security boundary. Hosted mode sends the hidden case to the
provider for grading; local mode runs inference on the configured machine. Multi-worker
persistence, authenticated per-user records, TLS termination, and account-level spending limits
belong in a deployment layer if the application is later operated as a public service.
