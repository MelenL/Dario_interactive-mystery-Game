# Validation status and manual evaluation

The application and tests were not executed during implementation. No model inference or
model-weight downloads were performed. Static source review does not establish runtime success.
Python AST parsing, TOML/JSON parsing, and Ruff static checks passed. Local documentation links
and the manual-only workflow configuration were also checked without importing the application.
Dependencies have compatible declared version ranges but a fully resolved environment has not
been installed or locked. Gradio rendering, provider access, hardware usage, generation quality,
and the offline tests remain to be validated explicitly.

## Offline contract tests

Install the package with the `dev` extra and run the commands in the README when ready.
`tests/conftest.py` blocks socket connections. Tests replace model calls with synthetic responses
and use deterministic embeddings for retrieval contracts. The tests exercise these boundaries:

- Story retrieval reaches the generator; the solution is absent from the public projection.
- Question answers require valid retrieved evidence references.
- Hint progression accounts for difficulty, previous hints, and unsuccessful hypotheses.
- Every required criterion must match in the current hypothesis, with real input quotes.
- Failed calls preserve the existing input state.
- Independent games do not inherit each other's dialogue or clues.
- Turn limits, empty inputs, schema errors, invalid references, and missing tokens are handled.

These contracts do not prove prompt-injection resistance or correct semantic judgement.
The manually triggered GitHub workflow runs the same checks; it contains no inference step.

## Live evaluation procedure

1. Start with one provider/model pair and record the model ID, provider, dependency versions,
   configuration, and date. Check access and quotas before generating several games.
2. Generate at least five cases across easy, medium, and hard settings. Inspect the private
   cases locally in a development session, not through a public UI endpoint.
3. Have a human reviewer check that the scene is consistent, the explanation is plausible, and
   the clues allow every rubric criterion to be inferred. Record rejected cases too.
4. For each case, prepare canonical yes/no questions, unrelated questions, unsupported questions,
   incomplete hypotheses, a correct paraphrase, and contradictory explanations.
5. Compare question answers against human labels. Evaluate correctness, abstention when evidence
   is absent, and whether the retriever selected the necessary facts.
6. Check hint novelty, supportedness, escalation, and whether a hint reveals the complete answer.
7. Check false wins, rejected correct paraphrases, quoted-evidence accuracy, and consistent handling
   of attempts to override the rubric or reveal the solution through a question.
8. Open two browser sessions and verify that cases, progress, and dialogue stay separate. Confirm
   that refresh, expiry, and a new case behave as documented.
9. Measure latency and inference usage separately from correctness. Keep the corpus of reviewed
   cases and failures when comparing a new prompt or model.

Report numerator/denominator counts and model settings with any results. Do not label a run
successful based only on valid JSON or attractive prose. No results are supplied here because
this evaluation has not been performed.
