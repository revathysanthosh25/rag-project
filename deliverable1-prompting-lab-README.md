# Week 2 — Deliverable 1: Prompting Lab

`prompting_lab.ipynb` — five prompt engineering patterns applied to real extraction and classification tasks, built as part of the Agentic Engineering Internship (Week 2: LLMs, Prompt Engineering & RAG).

## In plain terms

Large language models (LLMs) like the one powering ChatGPT don't just "know" how to do a task the way a calculator knows how to add — *how you ask* changes *how well it answers*. This notebook is a hands-on tour of five different ways to phrase a request to an LLM, each suited to a different kind of problem:

- Sometimes you can just ask directly and get a good answer.
- Sometimes it helps to show the model a few examples of what you want first, so it can copy the pattern.
- Sometimes the task is complex enough that the model needs to "show its work" before giving a final answer, the same way a person would talk through a tricky decision out loud.
- Sometimes you don't want a sentence back at all — you want clean, structured data (like a form with specific fields) that other code can use automatically.
- And sometimes the best results come from being very explicit about *who* the model should act as and *what format* you expect back.

Each of the five exercises below picks a realistic, everyday scenario — classifying a product review, triaging a support ticket, pulling an order out of a rambling email — and shows the same underlying skill: getting consistent, usable answers out of an LLM instead of vague or inconsistent ones.

## Technical overview

### Setup

- **Provider:** [Groq](https://groq.com) (OpenAI-compatible chat completions API), used in place of OpenAI directly due to a billing/card issue on the original provider. The prompting techniques themselves are provider-agnostic — the `ask()` helper function below is the only piece that's Groq-specific, and could be pointed back at OpenAI (or any OpenAI-compatible API) by swapping the client and model name.
- **Model:** `openai/gpt-oss-20b` (OpenAI's open-weight 20B model, served on Groq's infrastructure).
- **Key management:** the Groq API key is loaded from a local `.env` file via `python-dotenv` (`GROQ_API_KEY`), never hardcoded into the notebook.
- **Schema validation:** `pydantic` (`BaseModel`) is used for the structured-output patterns, to enforce a typed schema on the model's JSON responses rather than trusting raw text.

```python
import os
from dotenv import load_dotenv
from groq import Groq
from pydantic import BaseModel

load_dotenv(dotenv_path="/Users/revathy_santhosh/rag-project/.env")
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def ask(messages, model="openai/gpt-oss-20b"):
    response = client.chat.completions.create(model=model, messages=messages)
    return response.choices[0].message.content
```

### The five patterns

**1. Zero-shot prompting — sentiment classification**
A single instruction with no examples: classify a product review as Positive, Negative, or Mixed. Tests whether the model can perform the task from a plain description alone, including handling reviews with both positive and negative elements correctly (e.g. praising battery life while criticizing build quality → `Mixed`, not a forced binary choice).

**2. Few-shot prompting — support ticket categorization**
Three labeled examples (`Billing` / `Technical` / `General`) are given as prior conversation turns before the model is asked to classify a new, unlabeled message. This lets the model infer the classification boundary from examples rather than a written rule — useful when the category definitions are easier to demonstrate than to describe precisely. The test case (an unexpected invoice fee) is correctly classified as `Billing` without the word "charge" or "billing" appearing in a way that could be pattern-matched trivially.

**3. Chain-of-thought prompting — priority triage**
The model is instructed to reason through impact and urgency before committing to a final answer, formatted as `Priority: <Low|Medium|High>`. This is applied to a scenario requiring the model to weigh multiple factors at once (number of people affected, business-criticality, deadline proximity) rather than pattern-match on a single keyword — the kind of task where jumping straight to an answer is more error-prone than reasoning through it first.

**4. Structured output via Pydantic — ticket field extraction**
```python
class TicketExtraction(BaseModel):
    category: str
    priority: str
    summary: str
```
The model is instructed to return *only* valid JSON matching this schema, and the raw response is parsed and validated with `TicketExtraction.model_validate_json(raw)`. This is the foundation of reliable LLM-to-code integration: instead of parsing free text with regex or hoping the format is consistent, the response is validated against a typed contract and fails loudly if it doesn't match — which matters once this pattern is reused in the agent/tool-calling work in later weeks.

**5. Role-Context-Task-Format (RCTF) structured extraction — order parsing**
```python
class OrderRequest(BaseModel):
    sender_name: str
    company: str
    item: str
    quantity: int
    deadline: str
```
The system prompt is explicitly structured into four labeled sections — **Role** (who the model is acting as), **Context** (background on the kind of input it'll see), **Task** (what to do), **Format** (exact output shape) — rather than one unstructured block of instructions. Applied to a deliberately messy, informal customer email with the actual order details buried in small talk, the model correctly extracted all five fields (`sender_name`, `company`, `item`, `quantity`, `deadline`) into a validated `OrderRequest` object. This structuring technique produced the most reliable extraction of the five patterns on noisy, real-world-style input.

### What this demonstrates

Across the five patterns, the notebook covers the core prompt-engineering toolkit called for in the Week 2 learning objectives: zero-shot, few-shot, and chain-of-thought prompting; structured output / function-calling-style extraction with Pydantic; and explicit prompt structuring (Role/Context/Task/Format). The last two patterns in particular — typed, validated output from an LLM — are the direct precursor to the RAG pipeline and agent-tool patterns in the following deliverables, where downstream code needs to consume LLM output programmatically rather than display it as text.
