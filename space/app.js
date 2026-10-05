import { decide, embed, loadEmbedder } from './router.js?v=7';

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
const size = (m) => m.split(':')[1].toUpperCase();
const pct = (x) => `${Math.round(x * 100)}%`;
const secs = (x) => `${x < 10 ? x.toFixed(1) : Math.round(x)} s`;

// Each example button draws a different prompt from its pool on every click.
const EXAMPLES = {
  'an email': [
    'Write a short, polite email to my landlord asking when the broken heater in my apartment will be fixed.',
    "Draft an email to my manager asking to move Friday's one-on-one to next Tuesday afternoon.",
    'Write a friendly follow-up email to a recruiter I interviewed with last week, asking about next steps.',
    'Write an email to a customer apologizing that their order shipped two days late, and offer free shipping on their next purchase.',
    'Draft a short email declining a meeting invitation because of a scheduling conflict, and suggest two other times.',
    'Write an email to my team announcing that the office will be closed on Monday for maintenance.',
    'Write a thank-you email to a colleague who helped me prepare for a big presentation.',
    'Write an email to a professor asking whether there is still space in their machine learning course next term.',
  ],
  'a word problem': [
    'A train leaves at 3:15 pm and travels 210 km at 84 km/h. At what time does it arrive?',
    'A shirt costs $40 and is on sale for 25% off. Sales tax is 13%. What is the final price?',
    'Maya reads 18 pages a day. Her book has 432 pages. How many weeks will it take her to finish it?',
    'A recipe needs 3 eggs for every 2 cups of flour. How many eggs are needed for 7 cups of flour?',
    'Tom is three times as old as his daughter. In 12 years he will be twice as old as her. How old is Tom now?',
    'A tank holds 1,200 liters and drains at 45 liters per minute. How long until it is one quarter full?',
    'Four friends split a $96 dinner bill plus a 15% tip equally. How much does each person pay?',
    "A rectangle's length is 4 cm more than its width, and its perimeter is 56 cm. What is its area?",
  ],
  'multiple choice': [
    "Which gas makes up most of Earth's atmosphere?\nA. Oxygen\nB. Nitrogen\nC. Carbon dioxide\nD. Argon",
    'Which data structure gives constant-time average lookup by key?\nA. Linked list\nB. Binary search tree\nC. Hash table\nD. Stack',
    'Who wrote the novel Pride and Prejudice?\nA. Charlotte Brontë\nB. Jane Austen\nC. Mary Shelley\nD. George Eliot',
    'What is the derivative of x^3?\nA. x^2\nB. 3x\nC. 3x^2\nD. x^4 / 4',
    "Which organelle produces most of a cell's ATP?\nA. Nucleus\nB. Ribosome\nC. Golgi apparatus\nD. Mitochondrion",
    'What usually happens to demand for a normal good when incomes rise?\nA. It falls\nB. It rises\nC. It stays the same\nD. It drops to zero',
    'Which planet has the shortest year?\nA. Mercury\nB. Venus\nC. Mars\nD. Jupiter',
    'Which HTTP status code means the requested resource was not found?\nA. 200\nB. 301\nC. 404\nD. 500',
  ],
  'some code': [
    'Write a Python function that returns the n-th Fibonacci number using iteration.',
    'Write a Python function that checks whether a string is a palindrome, ignoring case and spaces.',
    'Write a SQL query that returns the five customers with the highest total order value.',
    'Write a JavaScript function that removes duplicate values from an array while keeping the original order.',
    'Write a Python function that merges two sorted lists into one sorted list without using sort().',
    'Write a Python function that counts how many times each word appears in a text file.',
    'Write a function that returns the second largest number in a list of integers.',
    'Write a bash command that lists the 10 largest files under the current directory.',
  ],
};

let router = null;
const gearPositions = {};

const randomFrom = (pool, avoid) => {
  const options = pool.filter((p) => p !== avoid);
  return options[Math.floor(Math.random() * options.length)];
};

// Gearbox: one stop per model, smallest to largest; the filled knob slides to the pick.
function gearbox(key, models, chosen) {
  const to = models.indexOf(chosen);
  const from = gearPositions[key] ?? to;
  gearPositions[key] = to;
  const stops = models.map((m) => `<div class="stop${m === chosen ? ' on' : ''}"><span class="knob"></span><span class="gear">${size(m)}</span></div>`).join('');
  return `<div class="gearbox" data-from="${from}" data-to="${to}"><div class="track"></div><span class="shifter"></span>${stops}</div>
    <div class="gearcaps"><span>smaller · faster</span><span>larger · more capable</span></div>`;
}

function shiftGears(root) {
  root.querySelectorAll('.gearbox').forEach((box) => {
    const shifter = box.querySelector('.shifter');
    shifter.style.setProperty('--pos', box.dataset.from);
    shifter.getBoundingClientRect();
    requestAnimationFrame(() => shifter.style.setProperty('--pos', box.dataset.to));
  });
}

// ---------- route a prompt ----------

function status(text) { $('status').textContent = text; }

function progress(event) {
  if (event.status === 'progress' && event.file?.endsWith('.onnx') && event.total) {
    const mb = (b) => Math.round(b / 1e6);
    status(`downloading the router's embedding model, first visit only · ${mb(event.loaded)} of ${mb(event.total)} MB · meanwhile, the replay tab works right away`);
  } else if (event.status === 'ready') {
    status('');
  }
}

function warmUp() {
  loadEmbedder(progress).catch((err) => status(`could not load the embedding model: ${err.message}`));
}

function targetHint(value) {
  if (value >= 97) return `keep ${value}% of the 9B's answer quality on average · almost always the 9B`;
  if (value <= 84) return `keep ${value}% of the 9B's answer quality on average · mostly smaller, faster models`;
  return `keep ${value}% of the 9B's answer quality on average · lower is faster, higher is safer`;
}

function reasonText(d) {
  const pick = size(d.model), big = size(d.largest), target = pct(d.target.tau);
  const lead = Math.round(d.target.tau * 100) !== d.quality
    ? `Your ${d.quality}% target uses the settings tuned for ${target}, rounding up to be safe. ` : '';
  if (d.reason === 'novel') {
    return `${lead}This prompt is unlike anything the router learned from, so it does not trust its own predictions and sends it to the ${big} to be safe.`;
  }
  const bar = `For a ${target} target the router needs at least ${pct(d.bar)} of the ${big}'s rate on each prompt (the tick on each bar)`;
  const unfamiliar = d.novelty === 'medium'
    ? ` The prompt is a little unlike what the router learned from, so the bar was raised from ${pct(1 - d.target.delta)} to ${pct(d.bar)}.` : '';
  if (d.reason === 'close-enough') {
    return `${lead}On prompts like this the ${pick} gives a good answer ${pct(d.predicted[d.model])} of the time, ${pct(d.relative[d.model])} as often as the ${big}. ${bar}, and the ${pick} is the fastest model that clears it.${unfamiliar}`;
  }
  const smaller = d.models.filter((m) => m !== d.largest);
  const best = smaller.reduce((a, b) => (d.relative[b] > d.relative[a] ? b : a), smaller[0]);
  return `${lead}No smaller model comes close enough on prompts like this: the best, the ${size(best)}, reaches ${pct(d.relative[best])} of the ${big}'s rate. ${bar}, so the ${big} answers.${unfamiliar}`;
}

function renderDecision(d, ms) {
  const big = d.largest;
  const saved = 1 - d.seconds[d.model] / d.seconds[big];
  const speed = d.model === big ? `the same model as always using the ${size(big)}` : `about ${pct(saved)} faster than always using the ${size(big)}`;
  const barAt = d.bar * d.predicted[big];
  const rows = d.models.map((m) => `
    <div class="qrow${m === d.model ? ' chosen' : ''}">
      <span class="qname">${size(m)}</span>
      <span class="bar"><span class="fill" style="width:${(d.predicted[m] * 100).toFixed(1)}%"></span>${d.reason === 'novel' ? '' : `<span class="tick" style="left:${(barAt * 100).toFixed(1)}%"></span>`}</span>
      <span class="num">${pct(d.predicted[m])}</span>
      <span class="num">${pct(d.relative[m])}</span>
      <span class="num">${secs(d.seconds[m])}</span>
    </div>`).join('');
  const warning = d.warning
    ? `<p class="reason note">A hard one: even the ${size(big)} gets prompts like this right only ${pct(d.predicted[big])} of the time.</p>` : '';
  const familiarity = { high: 'familiar', medium: 'somewhat unfamiliar', low: 'unfamiliar' }[d.novelty];
  $('result').innerHTML = `
    <div class="card">
      <div class="eyebrow">routed to</div>
      <div class="verdict"><span class="model">Qwen3.5 ${size(d.model)}</span><span class="num">${speed}</span></div>
      ${gearbox('route', d.models, d.model)}
      <p class="reason">${reasonText(d)}</p>
      ${warning}
      <div class="qtable">
        <div class="qrow qhead"><span>model</span><span>chance of a good answer</span><span></span><span class="r">vs ${size(big)}</span><span class="r">time</span></div>
        ${rows}
      </div>
      <p class="legend"><b>chance of a good answer</b> how often each model got similar test prompts right ·
        <b>tick</b> the lowest the router accepts for your target ·
        <b>time</b> estimated to answer on an Apple M5, including loading the model</p>
      <div class="meta">target ${d.quality}% · ${d.kind === 'choice' ? 'multiple-choice' : 'free-form'} prompt, ${familiarity} · decided in ${Math.round(ms)} ms in your browser</div>
    </div>
    ${answerNote(d)}`;
  $('result').querySelector('[data-goto]').addEventListener('click', () => showTab('replay'));
  shiftGears($('result'));
}

// This page has no language models behind it, so it stops at the decision.
function answerNote(d) {
  const prompt = $('prompt').value.trim();
  const quoted = prompt.length <= 70 && !/["\n\\$`]/.test(prompt) ? prompt : 'your prompt';
  return `
    <div class="answer-note">
      <div class="eyebrow">where is the answer?</div>
      <p class="reason">This page only makes the routing decision; there are no language models behind it. Installed on your
        own computer, ShiftAI hands the prompt to Qwen3.5 ${size(d.model)} through Ollama and prints its answer. To read real
        answers the models gave, open <button class="linklike" type="button" data-goto="replay">replay real answers</button>.</p>
      <pre class="cmd">pip install "shiftai-router[server] @ git+https://github.com/Jaswant06/shiftai-router"
shiftai setup
shiftai ask "${esc(quoted)}" --quality ${d.quality}</pre>
      <a class="small" href="https://github.com/Jaswant06/shiftai-router#run-it-locally" target="_blank" rel="noopener">full setup, including the Ollama models</a>
    </div>`;
}

async function route() {
  const prompt = $('prompt').value.trim();
  if (!prompt) { $('prompt').focus(); return; }
  const button = $('route');
  button.disabled = true;
  try {
    warmUp();
    await loadEmbedder();
    status('');
    const start = performance.now();
    const vec = await embed(prompt);
    const d = decide(router, prompt, vec, Number($('target').value));
    renderDecision(d, performance.now() - start);
  } catch (err) {
    status(`something went wrong: ${err.message}`);
  } finally {
    button.disabled = false;
  }
}

// ---------- replay real answers ----------

function grade(family, q) {
  if (family === 'Coding') return q >= 0.5 ? 'passes tests' : 'fails tests';
  if (family === 'Open-ended') return q > 0 ? `acceptable · p=${q.toFixed(2)}` : 'not acceptable';
  return q >= 0.5 ? 'correct' : 'wrong';
}

function answerMeta(item, a) {
  const energy = a.joules == null ? '' : ` · ${Math.round(a.joules)} J`;
  return `${grade(item.family, a.quality)} · ${secs(a.seconds)}${energy}`;
}

function stat(label, item, a) {
  return `<div class="stat"><div class="eyebrow">${label}</div>
    <div class="statline"><span class="model">${size(a.model)}</span><span class="num">${answerMeta(item, a)}</span></div></div>`;
}

function renderReplay() {
  const item = router.replay[Number($('pick').value)];
  const target = $('replay-target').value;
  const chosen = item.picks[target];
  const pick = item.answers.find((a) => a.model === chosen);
  const big = item.answers[item.answers.length - 1];

  let outcome;
  if (pick === big) {
    outcome = `At a ${target}% target the router keeps this prompt on the ${size(big.model)}. Lower the target to see if a smaller model would take it.`;
  } else {
    const time = 1 - pick.seconds / big.seconds;
    const energy = pick.joules != null && big.joules ? ` and ${pct(1 - pick.joules / big.joules)} less energy` : '';
    const worse = pick.quality < big.quality
      ? ` Here the smaller model did worse than the ${size(big.model)}: the target holds on average across all 760 test prompts, not on every single one.` : '';
    outcome = `On this prompt ShiftAI's pick took ${pct(time)} less time${energy} than the ${size(big.model)}.${worse}`;
  }
  const judged = item.family === 'Open-ended'
    ? '<p class="reason">Open-ended answers were graded by a judge model against the reference answer; p is the calibrated chance that an answer it accepted is really acceptable.</p>' : '';
  const reference = item.reference
    ? `<div class="reference"><span class="label">reference answer</span> &nbsp;${esc(item.reference)}</div>` : '';

  $('replay-summary').innerHTML = `
    <div class="card">
      <div class="eyebrow">${esc(item.family)} · test prompt</div>
      <blockquote class="prompt">${esc(item.prompt)}</blockquote>
      ${reference}
      ${gearbox('replay', item.answers.map((a) => a.model), chosen)}
      <div class="compare">${stat('shiftai picks', item, pick)}${stat(`always ${size(big.model)}`, item, big)}</div>
      <p class="reason outcome">${outcome}</p>
    </div>`;

  $('replay-answers').innerHTML = `<div class="eyebrow answers-title">what each model actually said · time and energy measured on an apple m5</div>${judged}` +
    item.answers.map((a) => `
      <details class="ans${a === pick ? ' chosen' : ''}"${a === pick ? ' open' : ''}>
        <summary><span class="ans-model">Qwen3.5 ${size(a.model)}</span>${a === pick ? '<span class="pick">shiftai’s pick</span>' : ''}<span class="num">${answerMeta(item, a)}</span></summary>
        <div class="ans-body">${a.html}</div>
      </details>`).join('');
  shiftGears($('replay-summary'));
}

function setUpReplay() {
  const families = [...new Set(router.replay.map((r) => r.family))];
  $('pick').innerHTML = families.map((f) => `<optgroup label="${esc(f)}">` +
    router.replay.map((r, i) => [r, i]).filter(([r]) => r.family === f).map(([r, i]) => {
      const text = r.prompt.replace(/\s+/g, ' ');
      return `<option value="${i}">${esc(text.length > 62 ? text.slice(0, 61) + '…' : text)}</option>`;
    }).join('') + '</optgroup>').join('');
  // Start on a prompt where routing pays off and the smaller model still gets it right.
  const start = router.replay.findIndex((r) => {
    const a = r.answers.find((x) => x.model === r.picks['90']);
    return a !== r.answers[r.answers.length - 1] && a.quality >= 0.5;
  });
  $('pick').value = String(Math.max(start, 0));
  renderReplay();
}

// ---------- wiring ----------

function slider(input, output, onChange) {
  const update = () => { output.textContent = `${input.value}%`; onChange?.(Number(input.value)); };
  input.addEventListener('input', update);
  update();
}

function showTab(name) {
  document.querySelectorAll('.tab').forEach((t) => {
    const on = t.dataset.tab === name;
    t.classList.toggle('active', on);
    t.setAttribute('aria-selected', on);
  });
  document.querySelectorAll('.panel').forEach((p) => { p.hidden = p.id !== `tab-${name}`; });
}

async function main() {
  document.querySelectorAll('.tab').forEach((t) => t.addEventListener('click', () => showTab(t.dataset.tab)));
  $('chips').innerHTML = Object.keys(EXAMPLES).map((k) => `<button class="chip" type="button">${k}</button>`).join('');
  $('chips').querySelectorAll('.chip').forEach((chip) => chip.addEventListener('click', () => {
    $('prompt').value = randomFrom(EXAMPLES[chip.textContent], $('prompt').value);
    route();
  }));
  // Start the one-time model download as soon as someone shows interest in routing.
  $('prompt').addEventListener('focus', warmUp, { once: true });
  $('chips').addEventListener('pointerenter', warmUp, { once: true });
  $('route').addEventListener('pointerenter', warmUp, { once: true });
  $('prompt').addEventListener('keydown', (e) => { if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) route(); });
  $('route').addEventListener('click', route);
  slider($('target'), $('target-out'), (v) => { $('target-hint').textContent = targetHint(v); });

  router = await (await fetch('data.json')).json();
  $('pick').addEventListener('change', renderReplay);
  $('shuffle').addEventListener('click', () => {
    const current = Number($('pick').value);
    const others = router.replay.map((_, i) => i).filter((i) => i !== current);
    $('pick').value = String(others[Math.floor(Math.random() * others.length)]);
    renderReplay();
  });
  setUpReplay();
  slider($('replay-target'), $('replay-target-out'), renderReplay);
}

main().catch((err) => status(`could not load the router: ${err.message}`));
