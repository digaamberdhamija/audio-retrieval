# AI usage

This document describes where AI is used in this project, both inside the product itself and as a development tool while building it.

## In short

- **The solution is my own design.** Before any code was written, I defined the retrieval strategy and chose the stack:
  - overlapping time windows of several lengths
  - joining overlapping matches separately for each length
  - reranking across lengths
  - hybrid search weighted towards exact keyword matches
  - Postgres with pgvector, behind a FastAPI service
- **An AI coding assistant sped up the work.** I used it to implement that design, refactor it, test it against the real system, and draft documentation under my direction.
- **The product itself uses AI models** for transcription, embeddings and reranking.

## AI inside the product

| Job | Model | Notes |
|---|---|---|
| Transcription, with speaker separation and word timings | Google Gemini 3.5 Transcribe | Any provider that returns speakers and word timings can replace it. |
| Embeddings for meaning search | Google Gemini Embedding 2 | Sits behind a single function and is planned to be replaced by a locally hosted model. |
| Reranking across window lengths | GTE-ModernColBERT-v1 (LightOn), a ColBERT model run locally | Loaded when the service starts, and runs on the GPU when one is available. |

The system does not generate answers or summaries. Every result is a passage of the original transcript, with its recording, timestamps and speakers.

## AI in development

**Tool:** Claude Code (Anthropic) in the Claude desktop app, using the Claude Opus 5.5 model.

### What it was used for

- **Implementation.** It built the ingestion and search service from my specification: the database schema and indexes, window cutting, hybrid search, joining, the reranker slot, de-duplication and the web API.
- **Refactoring.** It reorganised the code into routes, services and core modules, and moved every setting into one YAML file. It also rewrote the existing transcription script as a readable, asynchronous part of the service.
- **Verification.** It ran the service against the real database and the Gemini APIs, and exercised every flow with direct API calls. This surfaced practical issues, such as free-tier rate limits and a timestamp glitch in one transcript. I decided how each one should be handled.
- **Documentation.** It drafted the README, the agent instructions (AGENTS.md and CLAUDE.md) and the engineering notes from my outline, and I then edited them.
- **Fact-checking.** It confirmed model behaviour and external references before they went into the code or the docs.

### How I kept it on track

- **Working rules from the first prompt:**
  - keep code simple and human-readable
  - use async Python only, with no threads
  - keep comments minimal, with no design reasoning in them
  - put every setting in YAML
  - ask clarifying questions instead of assuming

  These rules now live in AGENTS.md, so any agent working on the project picks them up.
- **Decisions stayed with me.** Every open implementation detail came back to me as a question, and I made the call.
- **Checks ran against the real system,** with a real database, real API calls and real audio, not mocks.
- **I reviewed the generated code** and edited the code and docs directly where needed.
- **Secrets stayed out of the conversation.** The API key lives in a local environment file excluded from version control, and it never appeared in prompts or code.

## Examples of AI practices I use in my regular work

- **MCP servers for Jira and Bitbucket.** The assistant reads tickets, acceptance criteria and pull request discussions directly instead of me copying them in, and the work stays linked to its ticket.
- **Skills for database migrations.** A reusable skill guides schema changes with SQLAlchemy and Alembic, so every migration follows the same steps:
  1. Update the models.
  2. Generate the migration.
  3. Review it.
  4. Check that it upgrades and downgrades cleanly.
- **Skills for documentation, with the Confluence MCP server.** A documentation skill writes design documents that follow established design practices, and the Confluence MCP server publishes them straight to Confluence.
- **Claude/Codex plugins for end-to-end testing.** The assistant drives a real browser to click through user flows and check what actually renders, which catches problems unit tests miss.
- **Early context compaction.** Once a session passes half of the model's context window, it is compacted into a summary. Quality drops as the context fills up, and compacting early keeps the assistant focused on the current task and its instructions.
- **Keeping AGENTS.md and CLAUDE.md current.** Whenever the assistant learns something new about a project, such as a convention, a pitfall or a decision, it goes into these files. The next session, and every teammate's agent, then starts with that knowledge instead of rediscovering it.
