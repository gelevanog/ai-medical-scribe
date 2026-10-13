"""Command line: notewell serve | mcp | visits | transcribe | draft | codes ... | data ... | models ... | eval ... | calls"""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from notewell.config import Settings, get_settings

app = typer.Typer(help="Notewell: an AI medical scribe demo (synthetic data, not for clinical use).", no_args_is_help=True)
codes_app = typer.Typer(help="The official ICD-10-CM code list.", no_args_is_help=True)
data_app = typer.Typer(help="The synthetic visit dataset.", no_args_is_help=True)
models_app = typer.Typer(help="Local speech models and free cloud models.", no_args_is_help=True)
eval_app = typer.Typer(help="Evaluation runs (artifacts under results/).", no_args_is_help=True)
app.add_typer(codes_app, name="codes")
app.add_typer(data_app, name="data")
app.add_typer(models_app, name="models")
app.add_typer(eval_app, name="eval")
console = Console()

Provider = Annotated[str | None, typer.Option(help="fake | openrouter | openai_compatible")]
Model = Annotated[str | None, typer.Option(help="model id (OpenRouter ids must end in :free unless paid models are allowed)")]
Fallbacks = Annotated[str | None, typer.Option(help="comma-separated fallback model ids")]


def _settings(provider: str | None = None, model: str | None = None, fallbacks: str | None = None, **extra: object) -> Settings:
    settings = get_settings()
    update: dict[str, object] = dict(extra)
    if provider:
        update["llm_provider"] = provider
    if model:
        update["llm_model"] = model
    if fallbacks is not None:
        update["llm_fallback_models"] = [f.strip() for f in fallbacks.split(",") if f.strip()]
    return settings.model_copy(update=update) if update else settings


def _model(settings: Settings, tag: str, *, cache: bool | None = None):  # type: ignore[no-untyped-def]
    from notewell.llm.factory import build_chat_model, resilient

    return resilient(build_chat_model(settings), settings, tag=tag, cache=cache)


# ------------------------------------------------------------------------------------------------- server
@app.command()
def serve(
    host: Annotated[str, typer.Option()] = "127.0.0.1",
    port: Annotated[int, typer.Option()] = 8000,
    reload: Annotated[bool, typer.Option()] = False,
) -> None:
    """The API (and the built-in mock FHIR server at /mock-fhir) for the web UI."""
    import uvicorn

    uvicorn.run("notewell.api.app:create_default_app", factory=True, host=host, port=port, reload=reload)


@app.command()
def mcp(
    http: Annotated[bool, typer.Option(help="Streamable HTTP instead of stdio")] = False,
    host: Annotated[str, typer.Option()] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="HTTP port (endpoint /mcp)")] = 8766,
) -> None:
    """MCP server for Claude Code / Claude Desktop (PHI masking enforced server-side)."""
    from notewell.mcp_server import serve as serve_mcp

    serve_mcp(get_settings(), http=http, host=host, port=port)


# ------------------------------------------------------------------------------------------------- visits
@app.command()
def visits() -> None:
    """List the visits in the local store."""
    from notewell.services import build_services

    svc = build_services(get_settings())
    table = Table("id", "date", "specialty", "status", "title")
    for v in svc.visits():
        table.add_row(v.id, v.visit_date, v.specialty, v.status, v.title)
    console.print(table)


@app.command()
def draft(visit_id: str, provider: Provider = None, model: Model = None, fallbacks: Fallbacks = None) -> None:
    """Draft the note and codes for one visit and print the result with its flags."""
    from notewell.services import build_services

    settings = _settings(provider, model, fallbacks)
    svc = build_services(settings, llm=_model(settings, "cli"))
    visit = svc.draft(visit_id, "cli")
    note = visit.note
    assert note is not None
    for s in note.statements:
        flags = ", ".join(f.kind for f in s.flags)
        console.print(f"[bold]{s.section[:1].upper()}[/] {s.text}  [dim]{s.support}[/] [red]{flags}[/]")
    for m in note.medications:
        console.print(f"[cyan]Rx[/] {m.name} {m.dose} {m.frequency} ({m.status}) [red]{', '.join(f.kind for f in m.flags)}[/]")
    for c in note.icd10:
        console.print(f"[magenta]ICD[/] {c.code} {c.description} [red]{', '.join(f.kind for f in c.flags)}[/]")
    if note.em:
        console.print(f"[magenta]E/M[/] {note.em.code} (rule estimate {note.mdm.code if note.mdm else '-'})")
    console.print(f"[dim]model {note.model}, timings {note.timings}, masked {note.phi}[/]")


@app.command()
def mask(visit_id: str) -> None:
    """Print exactly what a cloud model would receive for a visit (the masked transcript)."""
    from notewell.services import build_services

    console.print(build_services(get_settings()).masked(visit_id).text())


@app.command()
def purge(days: Annotated[int | None, typer.Option(help="retention period")] = None) -> None:
    """Delete audio uploads and PHI vaults of signed visits older than the retention period."""
    from notewell.services import build_services

    settings = get_settings()
    count = build_services(settings, seed=False).store.purge(older_than_days=days or settings.retention_days)
    console.print(f"purged {count} uploads")


# ------------------------------------------------------------------------------------------------- codes
@codes_app.command("download")
def codes_download() -> None:
    """Download the official ICD-10-CM FY2027 code descriptions from CDC (SHA-256 verified)."""
    from notewell.codes.icd10 import FULL_URL, download_full

    path = download_full(get_settings().codes_dir)
    console.print(f"downloaded {FULL_URL}\n-> {path}")


@codes_app.command("search")
def codes_search(query: str, limit: int = 10) -> None:
    from notewell.codes.icd10 import load_codeset

    cs = load_codeset(str(get_settings().codes_dir))
    for code, score in cs.search(query, limit=limit):
        console.print(f"{code.dotted:9} {score:5.1f}  {code.title}")


@codes_app.command("validate")
def codes_validate(codes: list[str]) -> None:
    from notewell.codes.icd10 import load_codeset

    cs = load_codeset(str(get_settings().codes_dir))
    for c in codes:
        v = cs.validate(c)
        console.print(f"{v.code:9} {v.status:13} {v.title or v.message}")


# ------------------------------------------------------------------------------------------------- data
@data_app.command("validate")
def data_validate() -> None:
    """Check every visit file: evidence quotes verbatim, codes valid, identifiers present."""
    from notewell.dataset import load_cases
    from notewell.dataset.validate import validate_case

    settings = get_settings()
    from notewell.codes.icd10 import load_codeset

    cs = load_codeset(str(settings.codes_dir))
    bad = 0
    cases = load_cases(settings.dataset_dir)
    for case in cases:
        errors = validate_case(case, cs)
        if errors:
            bad += 1
            console.print(f"[red]FAIL[/] {case.id}: " + "; ".join(errors[:5]))
    stats = Counter(c.specialty for c in cases)
    console.print(f"{len(cases) - bad}/{len(cases)} valid; {dict(stats)}; facts {sum(len(c.gold.facts) for c in cases)}")
    if bad:
        raise typer.Exit(1)


@data_app.command("synth-audio")
def synth_audio(ids: Annotated[list[str] | None, typer.Argument()] = None, force: bool = False) -> None:
    """Speak every dialogue with two Kokoro voices (CPU) into data/audio (clean and noisy WAV + timings)."""
    from notewell.dataset import load_cases
    from notewell.speech.synth import Synthesizer

    settings = get_settings()
    synth = Synthesizer(settings.models_dir)
    for case in load_cases(settings.dataset_dir, ids=ids):
        if not force and (settings.audio_dir / f"{case.id}.json").exists():
            continue
        t = time.time()
        meta = synth.synthesize(case, settings.audio_dir)
        console.print(f"{case.id} {meta['voices']} {meta['duration_s']} s audio in {time.time() - t:.0f} s")


# ------------------------------------------------------------------------------------------------- models
@models_app.command("download")
def models_download(
    whisper: Annotated[list[str] | None, typer.Option(help="faster-whisper models")] = None,
    kokoro: bool = True,
) -> None:
    """Kokoro-82M (evaluation audio) and faster-whisper weights into the model cache."""
    from notewell.speech import download_models

    settings = get_settings()
    download_models(settings.models_dir, whisper=whisper or [settings.stt_model], kokoro=kokoro, log=console.print)


@models_app.command("free")
def models_free() -> None:
    """List OpenRouter's free models today (needs OPENROUTER_API_KEY)."""
    import httpx

    settings = get_settings()
    r = httpx.get(f"{settings.openrouter_base_url}/models", headers={"Authorization": f"Bearer {settings.openrouter_api_key}"}, timeout=30)
    r.raise_for_status()
    for m in sorted(r.json()["data"], key=lambda m: m["id"]):
        if m["id"].endswith(":free"):
            console.print(f"{m['id']:60} ctx {m.get('context_length')}")


# ------------------------------------------------------------------------------------------------- eval
@eval_app.command("smoke")
def eval_smoke(
    models: Annotated[str, typer.Option(help="comma-separated :free model ids")],
    visit: str = "pc-01",
    out: Path = Path("results/smoke.json"),
) -> None:
    """Draft one visit with each candidate free model: valid JSON, recall, codes, latency."""
    from notewell.codes.icd10 import load_codeset
    from notewell.dataset import load_cases
    from notewell.eval.note_eval import score_note
    from notewell.scribe.pipeline import Scribe

    settings = get_settings()
    case = load_cases(settings.dataset_dir, ids=[visit])[0]
    cs = load_codeset(str(settings.codes_dir))
    rows = []
    for model_id in [m.strip() for m in models.split(",") if m.strip()]:
        s = _settings("openrouter", model_id, "")
        v = case.visit()
        v.transcript = case.transcript()
        t = time.time()
        try:
            result = Scribe(_model(s, f"smoke:{model_id}"), cs, max_tokens=s.llm_max_tokens).draft(v)
            sc = score_note(case, result.note)
            row = {"model": model_id, "ok": True, "seconds": round(time.time() - t, 1), "fact_recall": round(sc.found / sc.facts, 3),
                   "statements": sc.statements, "unsupported": sc.unsupported, "icd": sc.icd_suggested[:3], "icd_top3": sc.icd_top3,
                   "em": sc.em_suggested, "negation_errors": len(sc.negation_errors), "tokens": result.note.usage}  # fmt: skip
        except Exception as exc:
            row = {"model": model_id, "ok": False, "seconds": round(time.time() - t, 1), "error": str(exc)[:300]}
        rows.append(row)
        console.print(row)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"visit": visit, "date": time.strftime("%Y-%m-%d"), "results": rows}, indent=1), encoding="utf-8")


@eval_app.command("notes")
def eval_notes(
    name: str,
    provider: Provider = None,
    model: Model = None,
    fallbacks: Fallbacks = None,
    transcripts: Annotated[str, typer.Option(help="'gold' or a directory of ASR transcripts")] = "gold",
    ids: Annotated[str | None, typer.Option(help="comma-separated visit ids")] = None,
    cache: bool = True,
) -> None:
    """Draft every visit with a model, score it against the gold annotations, check every payload for PHI."""
    from notewell.codes.icd10 import load_codeset
    from notewell.dataset import load_cases
    from notewell.eval.runner import RunConfig, run_notes

    settings = _settings(provider, model, fallbacks)
    cases = load_cases(settings.dataset_dir, ids=ids.split(",") if ids else None)
    cs = load_codeset(str(settings.codes_dir))
    if not cs.complete:
        console.print("[yellow]warning: the full ICD-10-CM list is not downloaded (notewell codes download)[/]")
    summary = run_notes(cases, _model(settings, f"eval:{name}", cache=cache), cs,
                        RunConfig(name, settings.results_dir / "runs" / name, transcripts), max_tokens=settings.llm_max_tokens)  # fmt: skip
    _print_summary(summary)


@eval_app.command("canary")
def eval_canary(
    provider: Provider = None,
    model: Model = None,
    fallbacks: Fallbacks = None,
    ids: Annotated[str | None, typer.Option()] = None,
    out: str = "canary",
) -> None:
    """Leak test: canary identities and an injected unregistered contact; no canary may reach any payload."""
    from notewell.codes.icd10 import load_codeset
    from notewell.dataset import load_cases
    from notewell.eval.runner import run_canary

    settings = _settings(provider, model, fallbacks)
    cases = load_cases(settings.dataset_dir, ids=ids.split(",") if ids else None)
    summary = run_canary(cases, _model(settings, "eval:canary", cache=True), load_codeset(str(settings.codes_dir)),
                         settings.results_dir / out, max_tokens=settings.llm_max_tokens)  # fmt: skip
    console.print({k: v for k, v in summary.items() if k != "cases"})


@eval_app.command("asr")
def eval_asr(
    model: Annotated[str, typer.Option(help="faster-whisper model")] = "small.en",
    condition: Annotated[str, typer.Option(help="clean | noisy")] = "clean",
    prompt: Annotated[bool, typer.Option(help="medical initial prompt")] = True,
    ids: Annotated[str | None, typer.Option()] = None,
    threads: int = 8,
) -> None:
    """WER, key-term recall and diarization role accuracy on the synthesized audio (local, no API calls)."""
    from notewell.eval.asr_runner import run_asr

    settings = get_settings()
    summary = run_asr(settings, model=model, condition=condition, prompt=prompt, ids=ids.split(",") if ids else None, threads=threads)
    console.print({k: v for k, v in summary.items() if k != "cases"})


@eval_app.command("phi")
def eval_phi() -> None:
    """PHI detection recall/over-masking on the dialogues and the two held-out challenge sets (local)."""
    from notewell.eval.phi_runner import run_phi

    console.print(run_phi(get_settings()))


@eval_app.command("guards")
def eval_guards(run: Annotated[str, typer.Option(help="a notes run whose notes are corrupted")] = "main") -> None:
    """Catch rate of the guards on deliberately corrupted notes (local)."""
    from notewell.eval.guard_eval import run_guard_eval

    console.print(run_guard_eval(get_settings(), run))


@eval_app.command("report")
def eval_report() -> None:
    """Build results/summary.json and results/report.md from the artifacts."""
    from notewell.eval.report import build_report

    path = build_report(get_settings())
    console.print(f"wrote {path}")


@app.command()
def calls(ledger: Path = Path("results/calls.jsonl")) -> None:
    """Summarize the call ledger: requests by model and status, tokens; verifies every model id ends in :free."""
    rows = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines() if line.strip()] if ledger.exists() else []
    by = Counter((r.get("requested_model"), r.get("status")) for r in rows)
    table = Table("model", "status", "requests")
    for (model, status), n in sorted(by.items(), key=lambda kv: (str(kv[0][0]), str(kv[0][1]))):
        table.add_row(str(model), str(status), str(n))
    console.print(table)
    non_free = {r.get("served_model") or r.get("requested_model") for r in rows if r.get("status") == "ok"} - {None}
    non_free = {m for m in non_free if not str(m).endswith(":free")}
    summary = {
        "requests": len(rows),
        "ok": sum(1 for r in rows if r.get("status") == "ok"),
        "input_tokens": sum(int(r.get("input_tokens") or 0) for r in rows),
        "output_tokens": sum(int(r.get("output_tokens") or 0) for r in rows),
        "non_free_models": sorted(non_free),
    }
    console.print(summary)
    (ledger.parent / "calls_summary.json").write_text(json.dumps({**summary, "by_model_status": {f"{m} {s}": n for (m, s), n in by.items()}}, indent=1), encoding="utf-8")


def _print_summary(summary: dict[str, object]) -> None:
    keys = ["visits", "fact_recall", "fact_recall_section_correct", "gold_matched_precision", "unsupported_rate", "negation_errors",
            "trap_errors", "medication_recall", "icd_top1", "icd_top3", "icd_top3_lenient", "em_exact", "em_within_one",
            "phi_leaks", "payloads_checked", "failures", "latency_s", "tokens_per_visit", "wall_minutes"]  # fmt: skip
    console.print({k: summary.get(k) for k in keys})


if __name__ == "__main__":
    app()
