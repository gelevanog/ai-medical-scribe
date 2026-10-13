"use client";

import Link from "next/link";
import { use, useCallback, useEffect, useMemo, useRef, useState } from "react";

import { CodesPanel } from "@/components/CodesPanel";
import { EyeIcon, FileIcon, LockIcon, ShieldIcon, SparkIcon } from "@/components/icons";
import { NoteEditor } from "@/components/NoteEditor";
import { Transcript } from "@/components/Transcript";
import { api, audioUrl, pdfUrl, type Section, SECTIONS, type Turn, type Visit } from "@/lib/api";
import { age, dateLabel, STATUS_LABEL, STATUS_STYLE } from "@/lib/format";

export default function VisitPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const [visit, setVisit] = useState<Visit | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState("");
  const [highlight, setHighlight] = useState<Set<number>>(new Set());
  const [showMasked, setShowMasked] = useState(false);
  const [maskedTurns, setMaskedTurns] = useState<Turn[] | null>(null);
  const [maskedCounts, setMaskedCounts] = useState<Record<string, number>>({});
  const [tab, setTab] = useState<"note" | "avs">("note");
  const [attested, setAttested] = useState(false);
  const [exported, setExported] = useState<{ server: string; resources: string[] } | null>(null);
  const [playing, setPlaying] = useState<number | null>(null);
  const audio = useRef<HTMLAudioElement>(null);

  useEffect(() => {
    api.visit(id).then(setVisit).catch((e: Error) => setError(e.message));
  }, [id]);

  const act = useCallback(async (label: string, fn: () => Promise<Visit>) => {
    setBusy(label);
    setError("");
    try {
      setVisit(await fn());
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy("");
    }
  }, []);

  const toggleMasked = useCallback(async () => {
    if (!showMasked && !maskedTurns) {
      try {
        const m = await api.masked(id);
        setMaskedTurns(m.turns);
        setMaskedCounts(m.counts);
      } catch (e) {
        setError((e as Error).message);
        return;
      }
    }
    setShowMasked((v) => !v);
  }, [id, maskedTurns, showMasked]);

  const note = visit?.note ?? null;
  const locked = visit?.status === "signed" || visit?.status === "exported";
  const openIssues = useMemo(() => {
    if (!note) return 0;
    const all = [...note.statements.flatMap((s) => s.flags), ...note.medications.flatMap((m) => m.flags), ...note.avs_flags];
    const codes = [...note.icd10, ...note.procedures, ...(note.em ? [note.em] : [])].filter((c) => c.accepted !== false).flatMap((c) => c.flags);
    return [...all, ...codes].filter((f) => f.severity === "error" && !f.acknowledged).length;
  }, [note]);
  const accepted = note ? SECTIONS.filter((s) => note.section_status[s] === "accepted").length : 0;
  const codesAccepted = note ? note.icd10.filter((c) => c.accepted).length : 0;
  const canSign = !!note && !locked && accepted === 4 && openIssues === 0 && codesAccepted > 0 && attested;

  if (error && !visit) return <div className="p-8 text-sm text-red-700">{error}</div>;
  if (!visit) return <div className="p-8 text-sm text-zinc-400">Loading…</div>;
  const p = visit.patient;
  const turns = showMasked && maskedTurns ? maskedTurns : visit.transcript?.turns ?? [];
  const actions = {
    edit: (sid: string, text: string) => act("Saving", () => api.edit(id, sid, text)),
    remove: (sid: string) => act("Removing", () => api.remove(id, sid)),
    add: (section: Section, text: string) => act("Adding", () => api.add(id, section, text)),
    acknowledge: (target: string, kind: string) => act("Saving", () => api.acknowledge(id, target, kind)),
    accept: (section: Section) => act("Accepting", () => api.accept(id, section)),
    reopen: (section: Section) => act("Reopening", () => api.reopen(id, section)),
  };

  return (
    <div className="flex h-screen flex-col">
      <header className="border-b border-line bg-white px-6 py-3">
        <div className="flex items-center gap-2 text-xs text-zinc-400">
          <Link href="/" className="hover:text-brand">Visits</Link>
          <span>/</span>
          <span>{visit.id}</span>
          {visit.synthetic && <span className="rounded bg-violet-50 px-1.5 font-medium text-violet-700">synthetic patient</span>}
        </div>
        <div className="mt-1 flex flex-wrap items-center gap-x-6 gap-y-2">
          <div>
            <h1 className="text-lg font-semibold leading-tight">{p.name}</h1>
            <div className="text-xs text-zinc-500">
              {age(p.dob) !== null && `${age(p.dob)} y · `}DOB {p.dob} · {p.sex} · MRN {p.mrn || "-"}
            </div>
          </div>
          <div className="text-xs text-zinc-500">
            <div className="font-medium text-zinc-800">{visit.title}</div>
            {dateLabel(visit.visit_date)} · {visit.summary.specialty_label} · {visit.visit_type === "new" ? "new patient" : "established"} · {visit.clinician.name}
          </div>
          <span className={`rounded-full px-2.5 py-0.5 text-xs font-medium ${STATUS_STYLE[visit.status]}`}>{STATUS_LABEL[visit.status]}</span>
          <div className="ml-auto flex items-center gap-2">
            {busy && <span className="animate-pulse text-xs text-zinc-500">{busy}…</span>}
            {visit.status === "new" && (
              <button onClick={() => act("Transcribing", () => api.transcribe(id))} className="rounded-md bg-ink px-3 py-1.5 text-sm font-medium text-white">
                Transcribe
              </button>
            )}
            {!locked && visit.transcript && (
              <button
                onClick={() => act("Drafting (masked transcript to the model)", () => api.draft(id))}
                className="flex items-center gap-1.5 rounded-md bg-brand px-3 py-1.5 text-sm font-medium text-white hover:bg-teal-800"
              >
                <SparkIcon className="h-4 w-4" /> {note ? "Re-draft" : "Draft note"}
              </button>
            )}
            {note && (
              <a href={pdfUrl(id)} target="_blank" rel="noreferrer" className="flex items-center gap-1.5 rounded-md px-3 py-1.5 text-sm text-zinc-700 ring-1 ring-line hover:bg-zinc-50">
                <FileIcon className="h-4 w-4" /> PDF
              </a>
            )}
          </div>
        </div>
        {error && <div className="mt-2 rounded-md bg-red-50 px-3 py-1.5 text-xs text-red-700">{error}</div>}
      </header>

      <div className="grid min-h-0 flex-1 grid-cols-[minmax(300px,1fr)_minmax(420px,1.45fr)_minmax(300px,0.95fr)]">
        {/* Transcript */}
        <section className="flex min-h-0 flex-col border-r border-line bg-white">
          <div className="flex items-center justify-between border-b border-line/70 px-4 py-2">
            <h2 className="text-sm font-semibold">Transcript</h2>
            <button
              onClick={toggleMasked}
              className={`flex items-center gap-1 rounded-full px-2.5 py-0.5 text-[11.5px] font-medium ${showMasked ? "bg-violet-600 text-white" : "text-violet-700 ring-1 ring-violet-300 hover:bg-violet-50"}`}
              title="Exactly what the cloud model receives"
            >
              <EyeIcon className="h-3.5 w-3.5" /> What the AI sees
            </button>
          </div>
          {visit.audio && (
            <div className="border-b border-line/70 px-3 py-2">
              <audio ref={audio} controls preload="none" src={audioUrl(id)} className="h-8 w-full" onTimeUpdate={(e) => setPlaying(e.currentTarget.currentTime)} />
            </div>
          )}
          {showMasked && (
            <div className="border-b border-violet-200 bg-violet-50 px-4 py-2 text-[11.5px] text-violet-800">
              <ShieldIcon className="mr-1 inline h-3.5 w-3.5" />
              Masked on this machine before any cloud call: {Object.entries(maskedCounts).map(([k, v]) => `${v} ${k.toLowerCase()}`).join(", ") || "nothing found"}.
              An egress check scans every outgoing request again and blocks it if an identifier remains.
            </div>
          )}
          <div className="min-h-0 flex-1 px-2 py-2">
            {turns.length ? (
              <Transcript
                turns={turns}
                highlight={highlight}
                masked={showMasked}
                playing={playing}
                onSeek={(t) => {
                  if (audio.current) {
                    audio.current.currentTime = t;
                    void audio.current.play();
                  }
                }}
              />
            ) : (
              <div className="p-4 text-sm text-zinc-400">No transcript yet.</div>
            )}
          </div>
          {visit.transcript && (
            <div className="border-t border-line/70 px-4 py-1.5 text-[11px] text-zinc-400">
              {visit.transcript.source === "asr" ? `ASR: ${visit.transcript.asr_model}` : "Reference transcript of the synthetic dialogue"} · {visit.transcript.turns.length} turns
            </div>
          )}
        </section>

        {/* Note */}
        <section className="flex min-h-0 flex-col">
          <div className="flex items-center gap-1 border-b border-line bg-white px-4 py-1.5">
            {(["note", "avs"] as const).map((t) => (
              <button key={t} onClick={() => setTab(t)} className={`rounded-md px-3 py-1 text-sm ${tab === t ? "bg-ink text-white" : "text-zinc-600 hover:bg-zinc-100"}`}>
                {t === "note" ? "SOAP note" : "After-visit summary"}
              </button>
            ))}
            {note && (
              <div className="ml-auto flex items-center gap-3 text-[11px] text-zinc-400">
                <span><span className="mr-1 inline-block h-2 w-2 rounded-full bg-emerald-500" />supported</span>
                <span><span className="mr-1 inline-block h-2 w-2 rounded-full bg-amber-500" />partial</span>
                <span><span className="mr-1 inline-block h-2 w-2 rounded-full bg-red-500" />unsupported</span>
              </div>
            )}
          </div>
          <div className="scroll-thin min-h-0 flex-1 overflow-y-auto px-4 py-3">
            {!note ? (
              <div className="rounded-lg border border-dashed border-line bg-white p-8 text-center text-sm text-zinc-500">
                No draft yet. <b>Draft note</b> masks the transcript, sends only the masked text to the model, restores identifiers here,
                and checks every statement against the transcript.
              </div>
            ) : tab === "note" ? (
              <>
                <div className="mb-2 text-[11.5px] text-zinc-500">
                  Hover a statement to see its evidence in the transcript. Drafted by <span className="font-mono">{note.model}</span>
                  {note.timings.total ? ` in ${note.timings.total.toFixed(1)} s` : ""}.
                </div>
                <NoteEditor note={note} locked={locked} actions={actions} onHover={(t) => setHighlight(new Set(t))} focused={null} />
              </>
            ) : (
              <div className="rounded-lg border border-line bg-white p-4">
                <p className="whitespace-pre-line text-[14px] leading-relaxed text-zinc-800">{note.after_visit_summary || "No summary drafted."}</p>
                {note.avs_flags.map((f, i) => (
                  <div key={i} className="mt-2 rounded bg-amber-50 px-2 py-1 text-xs text-amber-800">{f.message}</div>
                ))}
              </div>
            )}
          </div>
        </section>

        {/* Codes and sign-off */}
        <aside className="scroll-thin min-h-0 overflow-y-auto border-l border-line bg-white px-4 py-3">
          {note ? (
            <>
              <CodesPanel
                visitId={id}
                note={note}
                visitType={visit.visit_type}
                locked={locked}
                onCode={(code, a) => act("Saving", () => api.code(id, code, a))}
                onAdd={(code) => act("Adding", () => api.addCode(id, code))}
                onAck={(target, kind) => actions.acknowledge(target, kind)}
                onHover={(t) => setHighlight(new Set(t))}
              />
              <div className="mt-5 rounded-lg border border-line bg-zinc-50 p-3">
                <h3 className="flex items-center gap-1.5 text-sm font-semibold"><LockIcon className="h-4 w-4" /> Sign-off</h3>
                {locked ? (
                  <div className="mt-2 text-[12.5px] text-emerald-700">
                    Signed by {note.signed_by} on {note.signed_at ? new Date(note.signed_at).toLocaleString() : ""}. The note is locked.
                  </div>
                ) : (
                  <ul className="mt-2 space-y-1 text-[12.5px]">
                    <li className={accepted === 4 ? "text-emerald-700" : "text-zinc-500"}>{accepted === 4 ? "✓" : "○"} Sections accepted ({accepted}/4)</li>
                    <li className={openIssues === 0 ? "text-emerald-700" : "text-red-700"}>{openIssues === 0 ? "✓" : "○"} Open issues resolved ({openIssues} left)</li>
                    <li className={codesAccepted ? "text-emerald-700" : "text-zinc-500"}>{codesAccepted ? "✓" : "○"} At least one diagnosis accepted</li>
                    <li>
                      <label className="flex items-start gap-2 text-zinc-700">
                        <input type="checkbox" className="mt-0.5" checked={attested} onChange={(e) => setAttested(e.target.checked)} />
                        I reviewed this AI-drafted note against the visit and take responsibility for its content.
                      </label>
                    </li>
                  </ul>
                )}
                {!locked && (
                  <button
                    disabled={!canSign}
                    onClick={() => act("Signing", () => api.sign(id))}
                    className="mt-3 w-full rounded-md bg-ink py-2 text-sm font-medium text-white disabled:cursor-not-allowed disabled:opacity-35"
                  >
                    Sign note
                  </button>
                )}
                {locked && (
                  <button
                    onClick={async () => {
                      setBusy("Exporting");
                      try {
                        const r = await api.exportFhir(id);
                        setExported(r);
                        setVisit(await api.visit(id));
                      } catch (e) {
                        setError((e as Error).message);
                      } finally {
                        setBusy("");
                      }
                    }}
                    className="mt-3 w-full rounded-md bg-brand py-2 text-sm font-medium text-white"
                  >
                    {visit.status === "exported" ? "Export again (FHIR R4)" : "Export to EHR (FHIR R4)"}
                  </button>
                )}
                {exported && (
                  <div className="mt-2 rounded bg-white p-2 text-[11px] text-zinc-600 ring-1 ring-line">
                    <div className="font-medium text-zinc-700">{exported.resources.length} resources created at {exported.server}</div>
                    <div className="mt-1 font-mono text-[10.5px] leading-relaxed">{exported.resources.slice(0, 12).join(" · ")}</div>
                  </div>
                )}
              </div>
            </>
          ) : (
            <div className="text-sm text-zinc-400">Codes appear after the draft.</div>
          )}
        </aside>
      </div>
    </div>
  );
}
