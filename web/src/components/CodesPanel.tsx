"use client";

import { useState } from "react";

import { AlertIcon, CheckIcon } from "@/components/icons";
import { api, type Code, type Note } from "@/lib/api";
import { FLAG_LABEL } from "@/lib/format";

type Props = {
  visitId: string;
  note: Note;
  visitType: string;
  locked: boolean;
  onCode: (code: string, accepted: boolean | null) => Promise<void>;
  onAdd: (code: string) => Promise<void>;
  onAck: (target: string, kind: string) => Promise<void>;
  onHover: (turns: number[]) => void;
};

function CodeRow({ c, locked, onCode, onAck, onHover }: { c: Code; locked: boolean } & Pick<Props, "onCode" | "onAck" | "onHover">) {
  const errors = c.flags.filter((f) => f.severity === "error" && !f.acknowledged);
  const turns = c.evidence.filter((e) => e.verified).map((e) => e.turn);
  const quote = c.evidence.find((e) => e.verified && e.quote)?.quote;
  return (
    <li
      onMouseEnter={() => onHover(turns)}
      onMouseLeave={() => onHover([])}
      className={`rounded-md border px-2.5 py-2 ${c.accepted ? "border-emerald-200 bg-emerald-50/40" : c.accepted === false ? "border-line bg-zinc-50 opacity-60" : "border-line bg-white"}`}
    >
      <div className="flex items-start gap-2">
        <span className="rounded bg-ink px-1.5 py-0.5 font-mono text-[12px] font-semibold text-white">{c.code}</span>
        <div className="min-w-0 flex-1">
          <div className="text-[13px] font-medium leading-snug text-zinc-800">{c.description || "(no title)"}</div>
          <div className="mt-0.5 flex flex-wrap items-center gap-1.5 text-[10.5px]">
            {c.valid ? (
              <span className="rounded bg-emerald-50 px-1 font-medium text-emerald-700">valid · billable</span>
            ) : (
              <span className="rounded bg-red-50 px-1 font-medium text-red-700">not valid</span>
            )}
            {turns.length > 0 && <span className="font-mono text-zinc-400">{turns.map((t) => `#${t}`).join(" ")}</span>}
          </div>
        </div>
        {!locked && (
          <div className="flex shrink-0 gap-1">
            <button
              title="Accept"
              disabled={errors.length > 0}
              onClick={() => onCode(c.code, c.accepted ? null : true)}
              className={`rounded p-1 ${c.accepted ? "bg-emerald-600 text-white" : "text-emerald-700 ring-1 ring-emerald-300 hover:bg-emerald-50"} disabled:opacity-30`}
            >
              <CheckIcon className="h-3.5 w-3.5" />
            </button>
            <button
              title="Reject"
              onClick={() => onCode(c.code, c.accepted === false ? null : false)}
              className={`rounded px-1.5 text-xs ${c.accepted === false ? "bg-zinc-600 text-white" : "text-zinc-500 ring-1 ring-line hover:bg-zinc-100"}`}
            >
              ✕
            </button>
          </div>
        )}
      </div>
      {quote && <div className="mt-1.5 border-l-2 border-zinc-200 pl-2 text-[11.5px] italic text-zinc-500">&ldquo;{quote}&rdquo;</div>}
      {c.rationale && <div className="mt-1 text-[11.5px] text-zinc-500">{c.rationale}</div>}
      {c.flags.map((f, i) => (
        <div key={i} className={`mt-1 flex items-start gap-1 rounded px-1.5 py-0.5 text-[11px] ${f.acknowledged ? "bg-zinc-100 text-zinc-500" : f.severity === "error" ? "bg-red-50 text-red-700" : "bg-amber-50 text-amber-800"}`}>
          <AlertIcon className="mt-px h-3 w-3 shrink-0" />
          <span className="flex-1"><b>{FLAG_LABEL[f.kind] ?? f.kind}:</b> {f.message}</span>
          {!locked && !f.acknowledged && f.severity === "error" && (
            <button className="underline" onClick={() => onAck(`code:${c.code}`, f.kind)}>override</button>
          )}
        </div>
      ))}
    </li>
  );
}

export function CodesPanel({ visitId, note, visitType, locked, onCode, onAdd, onAck, onHover }: Props) {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<{ code: string; title: string }[]>([]);
  const mdm = note.mdm;
  void visitId;
  return (
    <div className="space-y-4">
      <div>
        <div className="mb-1.5 flex items-baseline justify-between">
          <h3 className="text-sm font-semibold">Diagnoses · ICD-10-CM</h3>
          <span className="text-[11px] text-zinc-400">FY2027, official CDC list</span>
        </div>
        <ul className="space-y-1.5">
          {note.icd10.map((c) => (
            <CodeRow key={c.code} c={c} locked={locked} onCode={onCode} onAck={onAck} onHover={onHover} />
          ))}
          {note.icd10.length === 0 && <li className="text-xs text-zinc-400">No suggestions.</li>}
        </ul>
        {!locked && (
          <div className="mt-2">
            <input
              value={query}
              onChange={async (e) => {
                setQuery(e.target.value);
                if (e.target.value.trim().length >= 3) setResults(await api.searchCodes(e.target.value).catch(() => []));
                else setResults([]);
              }}
              placeholder="Search the code list to add a code…"
              className="w-full rounded-md border border-line px-2.5 py-1.5 text-[13px] outline-none focus:border-brand"
            />
            {results.length > 0 && (
              <ul className="mt-1 max-h-48 overflow-y-auto rounded-md border border-line bg-white text-[12.5px] shadow-sm">
                {results.map((r) => (
                  <li key={r.code}>
                    <button
                      className="flex w-full gap-2 px-2.5 py-1.5 text-left hover:bg-brand-soft"
                      onClick={async () => {
                        await onAdd(r.code);
                        setQuery("");
                        setResults([]);
                      }}
                    >
                      <span className="font-mono font-semibold">{r.code}</span>
                      <span className="text-zinc-600">{r.title}</span>
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </div>
        )}
      </div>
      <div>
        <div className="mb-1.5 flex items-baseline justify-between">
          <h3 className="text-sm font-semibold">Visit level · E/M (CPT®)</h3>
          <span className="text-[11px] text-zinc-400">{visitType === "new" ? "new patient" : "established"}</span>
        </div>
        {note.em && (
          <ul>
            <CodeRow c={note.em} locked={locked} onCode={(code, a) => onCode(code, a)} onAck={onAck} onHover={onHover} />
          </ul>
        )}
        {mdm && (
          <div className="mt-2 rounded-md bg-zinc-50 px-2.5 py-2 text-[11.5px] text-zinc-600 ring-1 ring-line">
            <div className="mb-1 font-semibold text-zinc-700">Rule-based MDM estimate: {mdm.code} ({mdm.level})</div>
            <div className="grid grid-cols-3 gap-1.5">
              {(["problems", "data", "risk"] as const).map((k) => (
                <div key={k} className="rounded bg-white px-1.5 py-1 ring-1 ring-line">
                  <div className="text-[10px] uppercase tracking-wide text-zinc-400">{k}</div>
                  <div className="font-medium text-zinc-700">{mdm[k]}</div>
                </div>
              ))}
            </div>
            <ul className="mt-1.5 list-disc space-y-0.5 pl-4">
              {mdm.reasons.map((r) => (
                <li key={r}>{r}</li>
              ))}
            </ul>
          </div>
        )}
      </div>
      {note.procedures.length > 0 && (
        <div>
          <h3 className="mb-1.5 text-sm font-semibold">Procedures today · CPT®</h3>
          <ul className="space-y-1.5">
            {note.procedures.map((c) => (
              <CodeRow key={c.code} c={c} locked={locked} onCode={onCode} onAck={onAck} onHover={onHover} />
            ))}
          </ul>
        </div>
      )}
      <p className="text-[10.5px] leading-snug text-zinc-400">
        Suggestions for a certified coder. CPT® is a registered trademark of the AMA; descriptors shown are Notewell&apos;s own summaries.
      </p>
    </div>
  );
}
