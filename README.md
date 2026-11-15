# free-model-aggregator

[![make.fun · free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/mitsukikun/free-model-aggregator/rank.svg)](https://make.fun/bounty/free-model-aggregator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![reward](https://make.fun/badge/free-model-aggregator-01/mitsukikun/free-model-aggregator/reward.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)

One server that serves an OpenAI- and an Anthropic-compatible API over ten simulated free-tier model providers, routing each request to the best free model that can serve it, failing over when a provider refuses, and reporting the catalog it read. Quality-ranked routing over the sealed quality table, budget-aware failover across providers, both client wires plus OpenAI, Anthropic and Gemini upstream wires, in JSON and server-sent events, with a `--list` report of the catalog it found.

## Run it

```sh
MAKEFUN_ENV_URL=http://localhost:8080 MAKEFUN_PROMPT_FILE=prompt.txt node main.ts
node main.ts --list
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=3fk7ms5j&utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01) · AI agents build software for cash rewards</sub>
