// Screenshots of the review UI with headless Chrome (puppeteer-core).
//
//   PUPPETEER_FROM=<dir with node_modules/puppeteer-core> node docs/screenshots/capture.mjs <web-url> <api-url> [shots...]
//
// Shots: hero, masked, visits, signed, evaluation, audit. The API decides which notes are shown: the demo seeds the
// notes drafted by the free model in the main evaluation run (results/runs/main/notes).
import { createRequire } from "module";

const require = createRequire(process.env.PUPPETEER_FROM || import.meta.url);
const puppeteer = require("puppeteer-core");
const [web, api, ...requested] = process.argv.slice(2);
const shots = requested.length ? requested : ["hero", "masked", "visits", "signed", "evaluation", "audit"];
const out = process.env.OUT_DIR || new URL(".", import.meta.url).pathname;
const HERO_VISIT = process.env.HERO_VISIT || "cd-03";
const SIGN_VISIT = process.env.SIGN_VISIT || "pd-01";
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const browser = await puppeteer.launch({
  executablePath: process.env.CHROME || "/usr/bin/google-chrome",
  headless: "new",
  args: ["--no-sandbox", "--font-render-hinting=none"],
});

async function open(path, { width = 1280, height = 800, scale = 1.25 } = {}) {
  const page = await browser.newPage();
  await page.setViewport({ width, height, deviceScaleFactor: scale });
  await page.emulateMediaFeatures([{ name: "prefers-color-scheme", value: "light" }]);
  page.on("console", (msg) => {
    if (msg.type() === "error") console.log("page error:", msg.text());
  });
  await page.goto(web + path, { waitUntil: "networkidle0" });
  await sleep(600);
  return page;
}

async function hoverStatement(page, needle) {
  const handle = await page.evaluateHandle((n) => {
    const items = [...document.querySelectorAll("li")];
    return items.find((li) => li.innerText.toLowerCase().includes(n.toLowerCase())) || null;
  }, needle);
  const el = handle.asElement();
  if (!el) {
    console.log(`statement not found: ${needle}`);
    return;
  }
  await el.hover();
  await sleep(900);
}

async function call(method, path, body) {
  const r = await fetch(api + path, { method, headers: { "Content-Type": "application/json" }, body: body ? JSON.stringify(body) : undefined });
  if (!r.ok) throw new Error(`${method} ${path}: ${r.status} ${await r.text()}`);
  return r.json();
}

async function signAndExport(id) {
  let visit = await call("GET", `/api/visits/${id}`);
  if (visit.status === "signed" || visit.status === "exported") return;
  const note = visit.note;
  for (const s of note.statements) for (const f of s.flags) if (f.severity === "error") visit = await call("POST", `/api/visits/${id}/acknowledge`, { target: s.id, kind: f.kind });
  for (const [i, m] of visit.note.medications.entries())
    for (const f of m.flags) if (f.severity === "error") visit = await call("POST", `/api/visits/${id}/acknowledge`, { target: `med:${i}`, kind: f.kind });
  for (const c of [...visit.note.icd10, ...visit.note.procedures, visit.note.em].filter(Boolean)) {
    for (const f of c.flags) if (f.severity === "error") visit = await call("POST", `/api/visits/${id}/acknowledge`, { target: `code:${c.code}`, kind: f.kind });
  }
  for (const f of visit.note.avs_flags) if (f.severity === "error") visit = await call("POST", `/api/visits/${id}/acknowledge`, { target: "avs", kind: f.kind });
  for (const c of visit.note.icd10.slice(0, 2)) if (c.valid) await call("POST", `/api/visits/${id}/codes`, { code: c.code, accepted: true });
  if (visit.note.em) await call("POST", `/api/visits/${id}/codes`, { code: visit.note.em.code, accepted: true });
  for (const p of visit.note.procedures) if (p.valid) await call("POST", `/api/visits/${id}/codes`, { code: p.code, accepted: true });
  for (const s of ["subjective", "objective", "assessment", "plan"]) await call("POST", `/api/visits/${id}/sections/${s}/accept`);
  await call("POST", `/api/visits/${id}/sign`, { attestation: true });
}

for (const shot of shots) {
  if (shot === "hero") {
    const page = await open(`/visits/${HERO_VISIT}`);
    await hoverStatement(page, process.env.HERO_STATEMENT || "denies waking up");
    await page.screenshot({ path: `${out}/hero.png` });
    await page.close();
  } else if (shot === "masked") {
    const page = await open(`/visits/${HERO_VISIT}`);
    await page.evaluate(() => [...document.querySelectorAll("button")].find((b) => b.innerText.includes("What the AI sees"))?.click());
    await sleep(900);
    await page.screenshot({ path: `${out}/masked-transcript.png` });
    await page.close();
  } else if (shot === "visits") {
    const page = await open("/", { height: 900 });
    await page.screenshot({ path: `${out}/visits.png` });
    await page.close();
  } else if (shot === "signed") {
    await signAndExport(SIGN_VISIT);
    const page = await open(`/visits/${SIGN_VISIT}`);
    await page.evaluate(() => [...document.querySelectorAll("button")].find((b) => b.innerText.includes("Export to EHR"))?.click());
    await sleep(2500);
    await page.evaluate(() => {
      const aside = document.querySelector("aside.scroll-thin");
      if (aside) aside.scrollTop = aside.scrollHeight;
    });
    await sleep(500);
    await page.screenshot({ path: `${out}/signed-export.png` });
    await page.close();
  } else if (shot === "evaluation") {
    const page = await open("/evaluation", { height: 1000 });
    await page.screenshot({ path: `${out}/evaluation.png`, fullPage: true });
    await page.close();
  } else if (shot === "audit") {
    const page = await open("/audit", { height: 800 });
    await page.screenshot({ path: `${out}/audit-log.png` });
    await page.close();
  }
  console.log("captured", shot);
}
await browser.close();
