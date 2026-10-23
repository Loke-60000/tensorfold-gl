# tensorfold-gl

A fork of [TensorFold](https://github.com/ashhart/TensorFold) that adds per-request ablation.
Everything else is upstream; its README is in [README.upstream.md](README.upstream.md).

## What ablation is

A model's refusals are concentrated along a few directions inside it. Ablation measures those
directions once and removes them from each layer's output, so the model answers instead of
declining. No retraining or fine-tuning is involved; the weights stay as they are.

## Why per request

The usual way to get this is a second, "abliterated" copy of the model: either you switch
models and reload the weights each time, or you run both and need twice the memory.

Here one process serves both ids from one copy of the weights. The correction is a small
projection, applied only to the rows of requests that asked for the ablated id. Stock and
ablated requests share the same batches, so mixing them costs no reload and no extra
weights, just a few small kernel launches per layer for the ablated rows.

Typical uses are legitimate work a stock model often refuses:

- security research: reverse-engineering malware, explaining exploits, writing tooling for
  penetration tests and CTFs, red-teaming your own systems
- threat intelligence and incident response: summarising attacker reports, analysing
  phishing kits, reading forensic logs
- trust and safety work: labelling harmful content for moderation, building evaluation sets
  for safety filters
- studying refusal behaviour itself by sending the same prompt to both ids

Ablation removes refusals, not responsibility: the ablated id will answer requests the stock
model is right to refuse. Keep it behind authentication and only give it to people who need it.

## Usage

`TENSORFOLD_ABLATE_MODEL` names a second model id served from the same weights. A request
for that id reaches the engine with `Request.ablate` set; the served name and its aliases
don't. The id is listed in `/v1/models` and works for chat completions, completions,
Anthropic messages and Responses.

```sh
TENSORFOLD_ABLATE_MODEL=local-model-ablated tensorfold serve ~/models/my-model --name local-model
```

```sh
curl -fsS http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "local-model-ablated", "messages": [{"role": "user", "content": "Hello"}]}'
```

The flag reaches the backend as `Stream.ablate`; a family that reports `Info.ablate` applies
its own directions to those rows. No family in this branch does yet, so the server refuses
to start with `TENSORFOLD_ABLATE_MODEL` set until one does.

Ablated and stock requests never share prompt-cache states. An ablated request's key gets
its own namespace, folded into the first word the same way an image frame is.

## Tests

```sh
zig build test-server-cpu                    # ablation model: only its requests are ablated
zig build test-media-request-ownership-host  # Ablation cache: stock and ablated states stay apart
```
