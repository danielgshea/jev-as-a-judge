# Draft: subtle LangSmith references

_Review draft only. Not applied to the Notion page._

These are sentence-level replacements and additions to weave LangChain and LangSmith into the methodology, results, and conclusion. There is no new product section or callout.

## Methodology: opening paragraph

**Replace:**

> We built a weather agent with DeepAgents and gave it a single tool: Tavily, a web search tool.

**With:**

> We built a weather agent with DeepAgents and gave it a single tool: Tavily, a web search tool. We defined the five cases in a LangSmith dataset so that each evaluator could be tested against the same questions and expected behavior.

## Methodology: captured context paragraph

**Replace the first two sentences:**

> For each example, we captured the complete evaluation context once: the user’s question, the agent’s final answer, its search results, the tools it called, and the expected behavior. We then gave that identical context to each evaluator 100 times.

**With:**

> For each example, we captured the complete evaluation context once in LangSmith: the user’s question, the agent’s final answer, its search results, the tools it called, and the expected behavior. LangSmith’s evaluation SDK then replayed that identical context against each evaluator 100 times.

## Results: add after the quality-variance table

> Because every repetition was recorded in the same LangSmith experiment, we could compare the raw evaluator outputs alongside the aggregate variance, latency, and cost. That made it possible to inspect a charted result against the individual runs that produced it.

## Discussion: replace the annotation-queue sentences

**Replace:**

> In LangSmith, teams can send representative runs to an annotation queue, collect reviewer labels and scores, and compare that feedback with the judge’s output. The gaps show where to refine the evaluator’s rubric, prompt, and examples.

**With:**

> LangSmith closes that loop: teams can move representative experiment runs into an annotation queue, collect reviewer labels and scores, and compare that feedback with the judge’s output. The gaps show where to refine the evaluator’s rubric, prompt, and examples.

## Reproducibility: opening sentence

**Replace:**

> The published benchmark is `benchmark-jev-luna-terra-sonnet` (`6d08df72-c878-458c-b7c5-a7824ee6e721`), started at `2026-09-18T17:53:25Z`.

**With:**

> The reported output comes from the LangSmith experiment `benchmark-jev-luna-terra-sonnet` (`6d08df72-c878-458c-b7c5-a7824ee6e721`), which started at `2026-09-18T17:53:25Z`.

## Draft updates from the latest Notion comments

### Title

**Replace:** `Jev-as-a-Judge`

**With:** `Jev-as-a-Judge for Agent Evals`

This keeps the original hook while making the subject clear in search results and social previews.

### Opening link

The linked “LangChain + Jev implementation” is currently a private Notion page. Remove the sentence until there is a public destination, or replace it with a public GitHub example. Do not leave a reader-facing link that they cannot open.

### Why might Jev be more repeatable?

**Add after the first results summary:**

> This experiment cannot tell us why Jev’s scores varied less. One hypothesis is that the models are optimized for different kinds of output. TypeSafe describes Jev as a decision model trained to return calibrated probabilities and typed answers, while an autoregressive LLM judge generates text before the evaluator maps that output into a score. That difference may make Jev a better fit for this bounded evaluation task, but the result here is observational—not evidence that its training objective caused the lower variance.

### Methodology: simplify the expected-behavior explanation

**Replace the standalone paragraph beginning “Expected behavior can sound more precise than it is” with:**

> Each judge ran two evaluators against each of the five captured agent runs: `quality`, a continuous score that combines groundedness, search behavior, and usefulness; and `does_pass`, a binary decision. LangSmith’s evaluation SDK ran each evaluator 100 times per case. The expected behavior supplied the human-defined success criteria for both evaluators.

This addresses the placement concern while preserving the important point that the expected behavior is a human-defined standard.

## Comments already reflected in the current page

The resolved comments about expanding the code-evaluator example, validating evaluators against humans, using “autoregressive LLM,” adding typed-question examples, and explaining expected behavior are already reflected in the current Notion copy. No duplicate changes proposed.



# Updates!!!
 > “We built a weather agent with DeepAgents (https://www.langchain.com/deep-agents) and gave it Tavily, a web search tool. We defined the five cases in a LangSmith dataset so that
  > each evaluator could be tested against the same questions and expected behavior.”

  > “For each example, we captured the complete evaluation context once in LangSmith: the user’s question, the agent’s final answer, its search results, the tools it called, and the
  > expected behavior. LangSmith’s evaluation SDK then replayed that identical context against each evaluator 100 times.”

  > “Because every repetition was recorded in the same LangSmith experiment, we could compare the raw evaluator outputs alongside the aggregate variance, latency, and cost. That made
  > it possible to inspect a charted result against the individual runs that produced it.”

  > “LangSmith closes that loop: teams can move representative experiment runs into an annotation queue, collect reviewer labels and scores, and compare that feedback with the judge’s
  > output. The gaps show where to refine the evaluator’s rubric, prompt, and examples.”

# new updates!!!

The target agent was built with DeepAgents, an open source agent harness, given a simple prompt and access to a single web search tool. We then defined the test set as a LangSmith dataset so each evaluator ran against the same questions and expected behavior. The test set consists of five weather requests:

For each example in the dataset, we captured the weather agent’s response and stored the full output as a fixed example in LangSmith. Running the weather agent once and storing the output enabled us to isolate the evaluators as the sole source of potential variance. Using the LangSmith SDK, we invoked the evaluators on the fixed output with 100 repetitions.

LangSmith kept each repetition alongside its outputs, variance, latency, and cost inside the experiment. That gave us a way to trace each datapoint back to the individual runs behind it.