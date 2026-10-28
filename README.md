# tensorfold-gl

A fork of [TensorFold](https://github.com/ashhart/TensorFold) that adds per-request ablation: one process, one copy
of the weights, several model ids. Everything else is upstream v1.1.4; its README is in
[README.upstream.md](README.upstream.md).

## What ablation is

A model's refusals are concentrated along a few directions inside it. Ablation measures those
directions once and removes them from each layer's output, so the model answers instead of
declining. No retraining or fine-tuning is involved; the weights stay as they are.

## Why per request

The usual way to get this is a second, "abliterated" copy of the model: either you switch
models and reload the weights each time, or you run both and need twice the memory.

Here one process serves every id from one copy of the weights. The correction is a small
projection, applied only to the rows of requests that asked for an ablated id. Stock and
ablated requests share the same batches, so mixing them costs no reload and no extra
weights, just one small kernel launch per corrected writer and layer for those rows.

Typical uses are legitimate work a stock model often refuses:

- security research: reverse-engineering malware, explaining exploits, writing tooling for
  penetration tests and CTFs, red-teaming your own systems
- threat intelligence and incident response: summarising attacker reports, analysing
  phishing kits, reading forensic logs
- trust and safety work: labelling harmful content for moderation, building evaluation sets
  for safety filters
- studying refusal behaviour itself by sending the same prompt to several ids

Ablation removes refusals, not responsibility: an ablated id will answer requests the stock
model is right to refuse. Keep it behind authentication and only give it to people who need it.

## Identities

`TENSORFOLD_IDENTITIES` lists the ablated ids, `id=path` entries separated by commas (at most 32). Each path is a
safetensors file: `directions` [k, d] fp32 (orthonormal rows, k <= 4), `lam_attn` [k] and `lam_shared` [k] fp32
strengths, `first_layer` [1] int64. Every id is listed in `/v1/models` and works for chat completions, completions,
Anthropic messages and Responses; the served name and its aliases stay stock.

```sh
TENSORFOLD_IDENTITIES="local-model-open=/models/ablate/open.safetensors,local-model-mild=/models/ablate/mild.safetensors" \
  tensorfold serve ~/models/my-model --name local-model
```

```sh
curl -fsS http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "local-model-open", "messages": [{"role": "user", "content": "Hello"}]}'
```

`TENSORFOLD_ABLATE=path` with `TENSORFOLD_ABLATE_MODEL=id` names a single identity the short way;
`TENSORFOLD_ABLATE_SCALE` multiplies every strength (default 1). With no identity listed nothing runs and the
forward is unchanged.

## How it works

The request's identity (1..n in the list's order, 0 for the stock model) travels from the server through
`Request.ablate` to the lane's stream and the family's sequence. The Qwen3.8 Flash Next family
(`zig/src/families/flashnext`) packs every identity's directions and strengths into one device table and
uploads each row's identity with the token ids before a pass; `fn_ablate_rows` then subtracts
`R diag(lam) (R^T y)` from the token mixers' output projections and the shared expert's down projection for
the rows that name an identity, from that identity's `first_layer` up, before the hyper-connection write-back.
Rows with identity 0 are neither read nor written. The MTP draft head is untouched; the main forward verifies
its drafts.

Identities never share prompt-cache states: a request's cache key carries its identity in its first word, folded
in the same way an image frame is, so each id resumes its own kept state and never another's.

The two commits on top of v1.1.4's patch series are the recipe's patches 0018 and 0019 of
[Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold](https://github.com/Loke-60000/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold),
which serves this on one DGX Spark; `docs/ablation.md` there has the settings, the file format and the checks.

## Tests

```sh
zig build test-lane-api-host   # identities: the list and its errors; an identity's states never meet a stock one's nor another's
```
