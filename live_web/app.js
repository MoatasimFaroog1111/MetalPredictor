const $ = (id) => document.getElementById(id);
const money = new Intl.NumberFormat('en-US', {style:'currency', currency:'USD', minimumFractionDigits:2, maximumFractionDigits:2});
const pct = (logReturn) => `${((Math.exp(Number(logReturn)) - 1) * 100).toFixed(4)}%`;
const stamp = (value) => value ? new Date(value).toISOString().replace('.000Z','Z') : '—';

function direction(el, value) {
  const text = value || '—';
  el.textContent = text;
  el.className = `direction ${text === 'UP' ? 'up' : text === 'DOWN' ? 'down' : 'neutral'}`;
}

function setService(ok, label) {
  const pill = $('service-pill');
  pill.className = `status-pill ${ok ? 'ok' : 'bad'}`;
  pill.querySelector('b').textContent = label;
}

function applyForecast(data) {
  $('current-price').textContent = money.format(data.current_price_usd_per_kg);
  $('feature-time').textContent = `Feature hour: ${stamp(data.feature_timestamp_utc)} • Decision: ${stamp(data.decision_time_utc)}`;
  $('baseline-model').textContent = data.baseline_model;
  $('challenger-model').textContent = data.challenger_model;
  direction($('baseline-direction'), data.baseline_direction);
  direction($('challenger-direction'), data.challenger_direction);
  $('baseline-price').textContent = money.format(data.baseline_predicted_price_usd_per_kg);
  $('challenger-price').textContent = money.format(data.challenger_predicted_price_usd_per_kg);
  $('baseline-return').textContent = pct(data.baseline_log_return_1h);
  $('challenger-return').textContent = pct(data.challenger_log_return_1h);
  $('data-quality').textContent = data.data_quality;
  $('source-provider').textContent = data.source_provider;
  $('feed-compatibility').textContent = data.source_compatible_with_training ? 'TRAINING-COMPATIBLE' : 'UNVERIFIED CROSS-FEED';
}

function applyStatus(data) {
  $('edge-status').textContent = data.model?.edge_status || 'NOT_PROVEN';
  $('feature-count').textContent = data.model?.feature_count ?? '52';
  const tg = data.telegram || {};
  $('telegram-status').textContent = tg.notifications_enabled ? (tg.webhook_ready ? 'READY' : 'NOTIFY ONLY') : 'DISABLED';
}

function renderTable(history) {
  const body = $('history-body');
  body.replaceChildren();
  if (!history.length) {
    const tr = document.createElement('tr');
    const td = document.createElement('td');
    td.colSpan = 5;
    td.className = 'muted';
    td.textContent = 'لا توجد بيانات بعد.';
    tr.appendChild(td); body.appendChild(tr); return;
  }
  for (const row of history.slice(0, 20)) {
    const tr = document.createElement('tr');
    const cells = [
      stamp(row.feature_timestamp_utc),
      money.format(row.current_price_usd_per_kg),
      `${row.baseline_direction} ${pct(row.baseline_log_return_1h)}`,
      `${row.challenger_direction} ${pct(row.challenger_log_return_1h)}`,
      row.data_quality,
    ];
    cells.forEach((value, i) => {
      const td = document.createElement('td'); td.textContent = value;
      if (i === 2) td.className = row.baseline_direction === 'UP' ? 'up-text' : row.baseline_direction === 'DOWN' ? 'down-text' : '';
      if (i === 3) td.className = row.challenger_direction === 'UP' ? 'up-text' : row.challenger_direction === 'DOWN' ? 'down-text' : '';
      tr.appendChild(td);
    });
    body.appendChild(tr);
  }
}

function renderChart(history) {
  const svg = $('price-chart');
  const empty = $('empty-chart');
  svg.replaceChildren();
  if (history.length < 2) { empty.hidden = false; return; }
  empty.hidden = true;
  const rows = [...history].reverse().slice(-48);
  const actual = rows.map(r => Number(r.current_price_usd_per_kg));
  const predicted = rows.map(r => Number(r.baseline_predicted_price_usd_per_kg));
  const all = actual.concat(predicted).filter(Number.isFinite);
  const min = Math.min(...all), max = Math.max(...all), span = Math.max(max - min, max * 0.0005, 1e-9);
  const W = 800, H = 250, pad = 20;
  const points = (series) => series.map((v,i) => {
    const x = pad + i * ((W - 2*pad) / Math.max(series.length - 1, 1));
    const y = H - pad - ((v - min) / span) * (H - 2*pad);
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(' ');
  const line = (series, opacity, dash='') => {
    const node = document.createElementNS('http://www.w3.org/2000/svg','polyline');
    node.setAttribute('points', points(series)); node.setAttribute('fill','none');
    node.setAttribute('stroke','currentColor'); node.setAttribute('stroke-width','2');
    node.setAttribute('opacity',opacity); node.setAttribute('vector-effect','non-scaling-stroke');
    if (dash) node.setAttribute('stroke-dasharray',dash);
    return node;
  };
  svg.style.color = '#d8dee6';
  svg.appendChild(line(actual, '.95'));
  svg.appendChild(line(predicted, '.42', '6 6'));
}

async function getJson(url, allow404=false) {
  const response = await fetch(url, {headers:{'Accept':'application/json'}, cache:'no-store'});
  if (allow404 && response.status === 404) return null;
  if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
  return response.json();
}

async function refresh() {
  try {
    const [status, latest, history] = await Promise.all([
      getJson('/api/v1/status'),
      getJson('/api/v1/forecast/latest', true),
      getJson('/api/v1/forecast/history?limit=100'),
    ]);
    applyStatus(status);
    if (latest) applyForecast(latest);
    renderTable(history);
    renderChart(history);
    setService(true, 'النظام يعمل');
    $('last-refresh').textContent = `آخر تحديث: ${new Date().toISOString().replace('.000Z','Z')}`;
  } catch (error) {
    console.error(error);
    setService(false, 'تعذر الاتصال');
  }
}

$('refresh-btn').addEventListener('click', refresh);
refresh();
setInterval(refresh, 60_000);

if ('serviceWorker' in navigator) {
  window.addEventListener('load', () => navigator.serviceWorker.register('/sw.js', {scope:'/'}).catch(console.error));
}
