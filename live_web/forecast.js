"use strict";

const HORIZON_LABELS = {
  "4h": "توقع الأربع ساعات القادمة",
  "12h": "توقع الاثنتي عشرة ساعة القادمة",
  "1d": "توقع اليوم التالي",
  "2d": "توقع اليومين التاليين",
  "30d": "توقع الثلاثين يومًا التالية"
};

const $ = (id) => document.getElementById(id);

function horizonFromPath() {
  const parts = window.location.pathname.split("/").filter(Boolean);
  const value = (parts[1] || "4h").toLowerCase();
  return Object.hasOwn(HORIZON_LABELS, value) ? value : "4h";
}

function formatPrice(value) {
  return Number.isFinite(Number(value))
    ? Number(value).toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })
    : "—";
}

function formatPercent(value) {
  return Number.isFinite(Number(value)) ? `${(Number(value) * 100).toFixed(2)}%` : "—";
}

function formatSaudi(iso) {
  if (!iso) return "—";
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return "—";
  return new Intl.DateTimeFormat("ar-SA", {
    timeZone: "Asia/Riyadh",
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false
  }).format(date);
}

function firstKey(obj) {
  if (!obj || typeof obj !== "object") return "—";
  const keys = Object.keys(obj);
  return keys.length ? keys.join(" + ") : "—";
}

function setStatus(ok, text) {
  const pill = $("page-status");
  pill.classList.toggle("ok", ok);
  pill.classList.toggle("bad", !ok);
  pill.querySelector("b").textContent = text;
}

function setNav(horizon) {
  document.querySelectorAll("[data-horizon]").forEach((anchor) => {
    anchor.classList.toggle("active", anchor.dataset.horizon === horizon);
  });
}

function renderEvidence(payload) {
  const evidence = payload.evidence || {};
  const quality = evidence.latest_bar_quality || {};
  const admission = evidence.latest_bar_admission || {};

  $("coverage").textContent = formatPercent(quality.coverage_ratio);
  $("snapshots").textContent = quality.snapshot_count == null
    ? "—"
    : `${quality.snapshot_count} / ${quality.expected_snapshot_count}`;
  $("access-mode").textContent = firstKey(quality.access_mode_counts);
  $("freshness").textContent = firstKey(quality.freshness_status_counts);
  $("completed-bars").textContent = evidence.completed_forward_bar_count ?? "0";
  $("admitted-bars").textContent = evidence.admitted_forward_bar_count ?? "0";
  $("admission-reason").textContent = admission.reason || "WAITING_FOR_COMPLETED_BAR";

  const badge = $("admission-badge");
  const admitted = admission.admitted === true;
  badge.textContent = admitted ? "ADMITTED" : "COLLECTING";
  badge.classList.toggle("pass", admitted);
  badge.classList.toggle("wait", !admitted);

  const coverage = Number(quality.coverage_ratio);
  $("coverage").className = Number.isFinite(coverage) && coverage >= 0.9 ? "value-good" : "value-warn";
}

function renderSelection(payload) {
  const selection = payload.model_selection_evidence || {};
  $("forecast-method").textContent = payload.forecast_method || "random_walk_zero_return";
  $("selection-scope").textContent = selection.selection_scope || "—";
  $("candidate-pass-count").textContent = selection.candidate_gate_pass_count == null
    ? "غير مطبق"
    : String(selection.candidate_gate_pass_count);
  $("historical-confirmation").textContent = selection.historical_confirmation_authorized ? "AUTHORIZED" : "NOT_AUTHORIZED";
  $("selection-note").textContent = selection.note || "في Stage 3 لم يجتز أي Candidate بوابة التطوير، لذلك بقي Random Walk هو الـBaseline البحثي المحافظ ولم تُفتح Historical Confirmation.";
}

function clearForecastTiming() {
  $("reference-price").textContent = "—";
  $("reference-time").textContent = "لا توجد Bar حالية اجتازت بوابة القبول.";
  $("forecast-price").textContent = "—";
  $("forecast-target-time").textContent = "—";
  $("forecast-return").textContent = "—";
  $("bar-start").textContent = "—";
  $("bar-end").textContent = "—";
  $("last-snapshot").textContent = "—";
  $("target-start").textContent = "—";
  $("target-end").textContent = "—";
}

function renderAvailable(payload) {
  const reference = payload.reference || {};
  const target = payload.target || {};
  const forecast = payload.forecast || {};
  const banner = $("state-banner");
  banner.classList.add("available");
  banner.classList.remove("collecting");
  $("state-title").textContent = "Baseline متاح من Bar اجتازت بوابة الجودة";
  $("state-message").textContent = "التوقع المرجعي للفترة التالية يساوي آخر Close Mid المقبول وفق random_walk_zero_return؛ هذا ليس إثباتًا لميزة تنبؤية.";

  $("reference-price").textContent = formatPrice(reference.close_mid_usd_per_kg);
  $("reference-time").textContent = `آخر Snapshot مرصود: ${formatSaudi(reference.last_observed_snapshot_utc)} بتوقيت السعودية`;
  $("forecast-price").textContent = `${formatPrice(forecast.predicted_close_mid_usd_per_kg)} USD/kg`;
  $("forecast-target-time").textContent = formatSaudi(target.bar_end_utc);
  $("forecast-return").textContent = `${Number(forecast.predicted_change_pct || 0).toFixed(2)}%`;
  $("prediction-interval").textContent = forecast.prediction_interval_available ? "متاح" : "غير متاح — لم ينجح Candidate في Stage 3";

  $("bar-start").textContent = formatSaudi(reference.bar_start_utc);
  $("bar-end").textContent = formatSaudi(reference.bar_end_utc);
  $("last-snapshot").textContent = formatSaudi(reference.last_observed_snapshot_utc);
  $("target-start").textContent = formatSaudi(target.bar_start_utc);
  $("target-end").textContent = formatSaudi(target.bar_end_utc);
  $("source-provider").textContent = reference.source_provider || "BullionVault";
}

function renderCollecting(payload) {
  clearForecastTiming();
  const banner = $("state-banner");
  banner.classList.remove("available");
  banner.classList.add("collecting");
  $("state-title").textContent = "COLLECTING_EVIDENCE";
  if (payload.reason === "NO_COMPLETED_FORWARD_BAR") {
    $("state-message").textContent = "لم تكتمل بعد Bar أمامية لهذه الفترة. لن يتم تصنيع أو استكمال بيانات مفقودة.";
  } else {
    $("state-message").textContent = "آخر Bar مكتملة لم تجتز بوابة الجودة الصارمة، لذلك لا تنشر الصفحة رقمًا تنبؤيًا من هذه Bar.";
  }
}

function render(payload, horizon) {
  $("page-subtitle").textContent = `${HORIZON_LABELS[horizon]} • BullionVault authenticated forward observations`;
  $("forecast-horizon-label").textContent = HORIZON_LABELS[horizon];
  renderEvidence(payload);
  renderSelection(payload);
  if (payload.forecast_available) renderAvailable(payload);
  else renderCollecting(payload);
  $("last-refresh").textContent = `آخر تحديث: ${formatSaudi(new Date().toISOString())}`;
  setStatus(true, payload.forecast_available ? "Baseline available" : "Collecting evidence");
}

async function load() {
  const horizon = horizonFromPath();
  setNav(horizon);
  setStatus(true, "جاري التحميل");
  try {
    const response = await fetch(`/api/v1/research/multi-horizon-forecast/${encodeURIComponent(horizon)}`, {
      headers: { "Accept": "application/json" },
      cache: "no-store"
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const payload = await response.json();
    render(payload, horizon);
  } catch (error) {
    clearForecastTiming();
    $("state-title").textContent = "تعذر تحميل حالة البحث";
    $("state-message").textContent = String(error);
    setStatus(false, "خطأ في الاتصال");
  }
}

$("refresh-page").addEventListener("click", load);
load();
