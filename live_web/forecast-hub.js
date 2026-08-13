"use strict";

const horizons = ["4h", "12h", "1d", "2d", "30d"];
let active = "4h";

async function loadHorizon(key) {
  if (!horizons.includes(key)) return;
  active = key;
  document.querySelectorAll("[data-horizon]").forEach((button) => {
    button.classList.toggle("active", button.dataset.horizon === key);
  });
  const response = await fetch(`/api/v1/research/multi-horizon-forecast/${key}`, {cache: "no-store"});
  const payload = await response.json();
  document.getElementById("horizon-json").textContent = JSON.stringify(payload, null, 2);
}

document.querySelectorAll("[data-horizon]").forEach((button) => {
  button.addEventListener("click", () => loadHorizon(button.dataset.horizon));
});

loadHorizon(active);
