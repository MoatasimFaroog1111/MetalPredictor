const money = new Intl.NumberFormat('en-US', {
  style: 'currency',
  currency: 'USD',
  minimumFractionDigits: 2,
  maximumFractionDigits: 4,
});

const quantity = new Intl.NumberFormat('en-US', {
  minimumFractionDigits: 3,
  maximumFractionDigits: 6,
});

const saudiTime = new Intl.DateTimeFormat('ar-SA-u-ca-gregory-nu-latn', {
  timeZone: 'Asia/Riyadh',
  hour: '2-digit',
  minute: '2-digit',
  hourCycle: 'h23',
});

const $ = (id) => document.getElementById(id);

function ensureStyles() {
  if (document.querySelector('link[data-spread-profit-styles]')) return;
  const link = document.createElement('link');
  link.rel = 'stylesheet';
  link.href = '/static/components/spread-profit-simulator.css';
  link.dataset.spreadProfitStyles = 'true';
  document.head.appendChild(link);
}

function numberOrNull(value) {
  if (value === '' || value === null || value === undefined) return null;
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : null;
}

function formatTime(value) {
  return value ? saudiTime.format(new Date(value)) : '—';
}

function setText(id, value) {
  const element = $(id);
  if (element) element.textContent = value;
}

function renderSide(prefix, scenario, targetProfit) {
  const margin = Number(scenario.expected_margin_usd_per_kg);
  setText(`${prefix}-margin`, `${money.format(margin)} / kg`);

  const minQuantity = scenario.minimum_quantity_for_target_profit_kg;
  const minProfit = scenario.expected_profit_at_minimum_quantity_usd;
  if (minQuantity === null || minQuantity === undefined) {
    setText(`${prefix}-quantity`, 'لا يحقق ربحًا موجبًا بعد السبريد');
    setText(`${prefix}-profit`, '—');
    return;
  }

  setText(`${prefix}-quantity`, `${quantity.format(minQuantity)} kg`);
  let detail = `${money.format(minProfit)} متوقع عند أقل كمية لتحقيق هدف ${money.format(targetProfit)}`;
  if (scenario.current_entry_top_of_book_sufficient === true) {
    detail += ' • سيولة الدخول الحالية عند أفضل سعر تكفي لهذه الكمية';
  } else if (scenario.current_entry_top_of_book_sufficient === false) {
    detail += ' • سيولة أفضل سعر الحالية لا تكفي لهذه الكمية';
  }
  setText(`${prefix}-profit`, detail);
}

function renderModel(prefix, modelData, targetProfit) {
  setText(`${prefix}-model`, modelData.model);
  setText(`${prefix}-direction`, modelData.direction);
  setText(`${prefix}-mid`, money.format(modelData.predicted_mid_usd_per_kg));
  setText(`${prefix}-bid`, money.format(modelData.predicted_bid_usd_per_kg));
  setText(`${prefix}-ask`, money.format(modelData.predicted_ask_usd_per_kg));
  renderSide(`${prefix}-long`, modelData.long, targetProfit);
  renderSide(`${prefix}-short`, modelData.short, targetProfit);
}

function renderResult(data) {
  const reference = data.reference;
  const assumptions = data.assumptions;
  const quote = data.market_quote || null;

  setText('spread-current-time', formatTime(reference.current_price_time_utc));
  setText('spread-target-time', formatTime(reference.forecast_target_time_utc));
  setText('spread-current-mid', money.format(reference.current_mid_usd_per_kg));
  setText('spread-current-bid', money.format(reference.current_bid_usd_per_kg));
  setText('spread-current-ask', money.format(reference.current_ask_usd_per_kg));
  setText('spread-current-width', money.format(assumptions.current_spread_usd_per_kg));
  setText('spread-forecast-width', money.format(assumptions.forecast_spread_usd_per_kg));

  if (quote) {
    setText('spread-quote-source', `${quote.source_provider} • ${quote.security_id}`);
    setText('spread-quote-mode', quote.access_mode);
    setText('spread-bid-quantity', `${quantity.format(quote.best_bid_quantity_kg)} kg`);
    setText('spread-ask-quantity', `${quantity.format(quote.best_ask_quantity_kg)} kg`);
  } else {
    setText('spread-quote-source', 'Manual assumption');
    setText('spread-quote-mode', 'USER_ASSUMPTION');
    setText('spread-bid-quantity', '—');
    setText('spread-ask-quantity', '—');
  }

  renderModel('spread-baseline', data.baseline, assumptions.target_profit_usd);
  renderModel('spread-challenger', data.challenger, assumptions.target_profit_usd);

  $('spread-result').hidden = false;
  $('spread-error').hidden = true;
}

async function parseResponse(response) {
  if (!response.ok) {
    let detail = `HTTP ${response.status}`;
    try {
      const body = await response.json();
      if (body?.detail) detail = String(body.detail);
    } catch (_) {
      // Keep the HTTP fallback.
    }
    throw new Error(detail);
  }
  return response.json();
}

async function calculate(form) {
  const useBullionVault = form.elements.quoteSource.value === 'bullionvault';
  const common = {
    forecast_spread_usd_per_kg: numberOrNull(form.elements.forecastSpread.value),
    target_profit_usd: numberOrNull(form.elements.targetProfit.value) ?? 1,
    fixed_round_trip_cost_usd: numberOrNull(form.elements.fixedCost.value) ?? 0,
    quantity_step_kg: numberOrNull(form.elements.quantityStep.value) ?? 0.001,
    minimum_trade_quantity_kg: numberOrNull(form.elements.minimumQuantity.value) ?? 0.001,
  };

  if (useBullionVault) {
    const response = await fetch('/api/v1/research/bullionvault/spread-profit/latest', {
      method: 'POST',
      headers: {'Accept':'application/json', 'Content-Type':'application/json'},
      cache: 'no-store',
      body: JSON.stringify(common),
    });
    renderResult(await parseResponse(response));
    return;
  }

  const currentSpread = numberOrNull(form.elements.currentSpread.value);
  if (currentSpread === null) throw new Error('أدخل السبريد الحالي للوضع اليدوي.');
  const response = await fetch('/api/v1/research/spread-profit/latest', {
    method: 'POST',
    headers: {'Accept':'application/json', 'Content-Type':'application/json'},
    cache: 'no-store',
    body: JSON.stringify({
      current_spread_usd_per_kg: currentSpread,
      ...common,
    }),
  });
  renderResult(await parseResponse(response));
}

async function refreshBullionVaultStatus() {
  const status = $('spread-source-status');
  if (!status) return;
  status.textContent = 'جاري الاتصال بـ BullionVault…';
  try {
    const response = await fetch('/api/v1/research/bullionvault/quote', {
      headers: {'Accept':'application/json'},
      cache: 'no-store',
    });
    const data = await parseResponse(response);
    const quote = data.quote;
    status.textContent = [
      `${quote.source_provider} ${quote.security_id}`,
      `Bid ${money.format(quote.best_bid_usd_per_kg)}`,
      `Ask ${money.format(quote.best_ask_usd_per_kg)}`,
      `Spread ${money.format(quote.spread_usd_per_kg)}`,
      quote.access_mode,
    ].join(' • ');
  } catch (cause) {
    status.textContent = `تعذر جلب BullionVault الآن: ${cause instanceof Error ? cause.message : 'خطأ غير معروف'}`;
  }
}

function syncSourceMode(form) {
  const manual = form.elements.quoteSource.value === 'manual';
  const manualField = $('spread-manual-current-field');
  const currentInput = form.elements.currentSpread;
  if (manualField) manualField.hidden = !manual;
  currentInput.disabled = !manual;
  currentInput.required = manual;
  setText(
    'spread-source-help',
    manual
      ? 'الوضع اليدوي يستخدم Spread تدخله أنت ولا يستدعي BullionVault.'
      : 'BullionVault Read-Only: يتم جلب Bid/Ask وعمق أفضل سعر، ولا توجد أي صلاحية لإرسال أوامر.',
  );
}

export function mountSpreadProfitSimulator() {
  ensureStyles();

  const openButton = $('spread-simulator-open');
  const dialog = $('spread-simulator-dialog');
  const closeButton = $('spread-simulator-close');
  const form = $('spread-simulator-form');

  if (!openButton || !dialog || !closeButton || !form) return;

  syncSourceMode(form);
  form.elements.quoteSource.addEventListener('change', () => syncSourceMode(form));

  openButton.addEventListener('click', () => {
    if (typeof dialog.showModal === 'function') dialog.showModal();
    else dialog.setAttribute('open', '');
    if (form.elements.quoteSource.value === 'bullionvault') refreshBullionVaultStatus();
  });

  closeButton.addEventListener('click', () => dialog.close());

  dialog.addEventListener('click', (event) => {
    if (event.target === dialog) dialog.close();
  });

  form.addEventListener('submit', async (event) => {
    event.preventDefault();
    const submit = $('spread-simulator-calculate');
    const error = $('spread-error');
    submit.disabled = true;
    submit.textContent = 'جاري الحساب…';
    try {
      await calculate(form);
    } catch (cause) {
      error.textContent = cause instanceof Error ? cause.message : 'تعذر إجراء الحساب.';
      error.hidden = false;
      $('spread-result').hidden = true;
    } finally {
      submit.disabled = false;
      submit.textContent = 'احسب السيناريو';
    }
  });
}
