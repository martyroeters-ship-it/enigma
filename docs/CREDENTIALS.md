# Credentials (all optional)

Enigma runs its full research cycle with **no credentials**: research uses keyless public
providers (arXiv, OpenAlex, Crossref, GitHub, Wikipedia), market data uses Binance public
archives (data.binance.vision) and OKX public funding history, and every LLM-dependent step has a
deterministic fallback that is recorded as such in the decision log. Keys only add capability.
Values are read from the environment at call time and are never stored, logged, or printed;
`enigma credentials` shows which are configured.

| Purpose | Environment variable(s) | Effect when set |
|---|---|---|
| Model provider: OpenAI | `OPENAI_API_KEY` (optional `OPENAI_BASE_URL`) | Enables model routes on OpenAI |
| Model provider: Anthropic | `ANTHROPIC_API_KEY` | Enables Claude routes |
| Model provider: Google | `GEMINI_API_KEY` or `GOOGLE_API_KEY` | Enables Gemini routes |
| Model provider: OpenRouter | `OPENROUTER_API_KEY` | Enables OpenRouter routes |
| Model provider: Groq / Together / DeepSeek / Mistral | `GROQ_API_KEY`, `TOGETHER_API_KEY`, `DEEPSEEK_API_KEY`, `MISTRAL_API_KEY` | Enables those routes |
| Local models (free) | `OLLAMA_HOST`, `VLLM_BASE_URL`, `LMSTUDIO_BASE_URL` | Uses operator-hosted models at zero cost |
| **Model spend ceiling** | `ENIGMA_MAX_DAILY_API_SPEND_USD` | Global daily cap; **default 0 = no paid model calls even if keys are set** |
| Search: OpenAlex | `OPENALEX_API_KEY` | Most useful search key: keyless OpenAlex shares a ~100 requests/day allowance per IP, and a depth-1 research run uses ~10 |
| Polite-pool contact | `ENIGMA_CONTACT_EMAIL` | Contact address sent to OpenAlex and Crossref |
| Search: Semantic Scholar | `SEMANTIC_SCHOLAR_API_KEY` | Adds Semantic Scholar (keyless access is rate limited, 429) |
| Search: GitHub | `GITHUB_TOKEN` | Raises GitHub search rate limits (works unauthenticated) |
| Structured decisions (evaluated, not integrated) | `TYPESAFE_API_KEY` | Official TypeSafe Jev (`api.typesafe.ai` only). See `docs/technology/jev-typesafe-assessment.md`. Do **not** use jevai.net. |
| Search: web | `BRAVE_SEARCH_API_KEY`, `TAVILY_API_KEY`, `SERPAPI_API_KEY` | Adds general web search providers |

Where models are used once a key and a non-zero spend cap are present: the Quant Research
agent's hypothesis critique (recorded as an unverified AI interpretation) and the Technology
Innovation model benchmark. Validation never uses a model
(import-linter contract `enigma.validation` must not import `enigma.gateway`).

Paid market data, paid infrastructure, and exchange trading keys are not used and must not be
added: M1 performs no trading of any kind.
