// The ShiftAI router, ported from src/shiftai (features.py, predictor.py,
// policy.py, router.py) so it runs entirely in the browser. The embedding
// model is the same nomic-embed-text, loaded with Transformers.js.

export const EMBED_MODEL = 'nomic-ai/nomic-embed-text-v1.5';
const TRANSFORMERS = 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1';
const EMBED_PREFIX = 'classification: ';
const MAX_CHARS = 4000;
const CHARS_PER_TOKEN = 4;
const WARN_BELOW = 0.5;
// Two or more lines that start like "A. ..." or "(B) ..." mean the prompt lists options.
const OPTION_LINE = /^\s*\(?[A-H][.)]\s+\S/gm;

const codePoints = (text) => Array.from(text);

export function promptKind(text) {
  return (text.match(OPTION_LINE) || []).length >= 2 ? 'choice' : 'open';
}

let extractor = null;

export function loadEmbedder(onProgress) {
  if (!extractor) {
    extractor = import(TRANSFORMERS)
      .then(({ pipeline }) => pipeline('feature-extraction', EMBED_MODEL, { dtype: 'fp16', progress_callback: onProgress }))
      .catch((err) => { extractor = null; throw err; });
  }
  return extractor;
}

export async function embed(text) {
  const fx = await loadEmbedder();
  const input = EMBED_PREFIX + codePoints(text).slice(0, MAX_CHARS).join('');
  const out = await fx(input, { pooling: 'mean', normalize: true });
  return out.data;
}

function nearest(clusters, vec) {
  let best = 0, sim = -Infinity;
  clusters.centroids.forEach((c, i) => {
    let s = 0;
    for (let d = 0; d < c.length; d++) s += c[d] * vec[d];
    if (s > sim) { sim = s; best = i; }
  });
  return [best, sim];
}

export function pickTarget(targets, quality) {
  const tau = quality > 1 ? quality / 100 : quality;
  const taus = Object.keys(targets).map(Number).sort((a, b) => a - b);
  const t = taus.find((x) => x >= tau - 1e-9) ?? taus[taus.length - 1];
  return { tau: t, ...targets[String(t)] };
}

export function estimateSeconds(router, model, prompt, loaded = new Set()) {
  const cost = router.profile[model];
  const outTokens = router.expected_tokens[model]?.[promptKind(prompt)] ?? 256;
  const promptTokens = codePoints(prompt).length / CHARS_PER_TOKEN;
  let seconds = loaded.has(model) ? 0 : cost.load_s;
  if (cost.prompt_tps > 0) seconds += promptTokens / cost.prompt_tps;
  if (cost.gen_tps > 0) seconds += outTokens / cost.gen_tps;
  return seconds;
}

// Same rule as policy.choose: the cheapest model whose predicted quality is at
// least (1 - delta) times the largest model's. Unfamiliar prompts play safer.
export function decide(router, prompt, vec, quality = 90) {
  const models = router.ladder.filter((m) => m in router.profile);
  const largest = models[models.length - 1];
  const target = pickTarget(router.targets, quality);
  const predictor = router.predictors[String(target.clusters)];
  const kind = promptKind(prompt);
  const clusters = predictor.kinds[kind];

  const predicted = {};
  let novelty = 'low', similarity = null;
  if (clusters) {
    const [i, sim] = nearest(clusters, vec);
    models.forEach((m) => { predicted[m] = clusters.accuracy[i][predictor.models.indexOf(m)]; });
    similarity = sim;
    novelty = sim < clusters.low ? 'low' : sim < clusters.medium ? 'medium' : 'high';
  } else {
    models.forEach((m) => { predicted[m] = 0; });
  }

  const seconds = {};
  models.forEach((m) => { seconds[m] = estimateSeconds(router, m, prompt); });

  let delta = target.delta, model = largest, reason;
  if (quality >= 100) {
    reason = 'max';
  } else if (novelty === 'low') {
    reason = 'novel';
  } else {
    if (novelty === 'medium') delta /= 2;
    const bar = (1 - delta) * predicted[largest];
    reason = 'none-cheaper';
    for (const m of [...models].sort((a, b) => seconds[a] - seconds[b])) {
      if (predicted[m] >= bar) {
        model = m;
        reason = m === largest ? 'none-cheaper' : 'close-enough';
        break;
      }
    }
  }

  const reference = Math.max(predicted[largest], 1e-9);
  const relative = Object.fromEntries(models.map((m) => [m, predicted[m] / reference]));
  return {
    model, reason, quality, kind, novelty, similarity, models, largest, target, delta,
    bar: 1 - delta, predicted, relative, seconds,
    warning: predicted[largest] < WARN_BELOW,
  };
}
