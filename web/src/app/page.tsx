"use client";

import Link from "next/link";
import { useCallback, useEffect, useMemo, useState } from "react";

import { NewVisitDialog } from "@/components/NewVisitDialog";
import { MicIcon, PlusIcon } from "@/components/icons";
import { api, type VisitSummary } from "@/lib/api";
import { age, clock, dateLabel, STATUS_LABEL, STATUS_STYLE } from "@/lib/format";

const FILTERS = ["all", "primary_care", "cardiology", "dermatology", "pediatrics", "mental_health"] as const;
const FILTER_LABEL: Record<string, string> = {
  all: "All",
  primary_care: "Primary care",
  cardiology: "Cardiology",
  dermatology: "Dermatology",
  pediatrics: "Pediatrics",
  mental_health: "Mental health",
};

export default function VisitsPage() {
  const [visits, setVisits] = useState<VisitSummary[] | null>(null);
  const [error, setError] = useState("");
  const [filter, setFilter] = useState<(typeof FILTERS)[number]>("all");
  const [open, setOpen] = useState(false);

  const load = useCallback(() => {
    api.visits().then(setVisits).catch((e: Error) => setError(e.message));
  }, []);
  useEffect(load, [load]);

  const shown = useMemo(() => (visits ?? []).filter((v) => filter === "all" || v.specialty === filter), [visits, filter]);
  const counts = useMemo(() => {
    const c: Record<string, number> = {};
    for (const v of visits ?? []) c[v.status] = (c[v.status] ?? 0) + 1;
    return c;
  }, [visits]);

  return (
    <div className="mx-auto max-w-6xl px-8 py-8">
      <header className="mb-6 flex items-end justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Visits</h1>
          <p className="mt-1 text-sm text-zinc-500">
            Record or upload a visit; Notewell transcribes it on this machine, drafts a SOAP note from the masked transcript and
            waits for your review and signature.
          </p>
        </div>
        <button
          onClick={() => setOpen(true)}
          className="flex shrink-0 items-center gap-2 whitespace-nowrap rounded-md bg-brand px-3.5 py-2 text-sm font-medium text-white shadow-sm hover:bg-teal-800"
        >
          <MicIcon className="h-4 w-4" /> New visit
        </button>
      </header>

      <div className="mb-4 grid grid-cols-4 gap-3">
        {[
          ["To review", (counts.drafted ?? 0) + (counts.transcribed ?? 0), "drafted or waiting for a draft"],
          ["Signed", counts.signed ?? 0, "locked, ready to export"],
          ["Exported", counts.exported ?? 0, "sent to the EHR (FHIR)"],
          ["Visits", visits?.length ?? 0, "all synthetic"],
        ].map(([label, value, hint]) => (
          <div key={String(label)} className="rounded-lg border border-line bg-white px-4 py-3">
            <div className="text-xs font-medium uppercase tracking-wide text-zinc-500">{label}</div>
            <div className="mt-1 text-2xl font-semibold tabular-nums">{value}</div>
            <div className="text-xs text-zinc-400">{hint}</div>
          </div>
        ))}
      </div>

      <div className="mb-3 flex flex-wrap gap-1.5">
        {FILTERS.map((f) => (
          <button
            key={f}
            onClick={() => setFilter(f)}
            className={`rounded-full px-3 py-1 text-[13px] ${filter === f ? "bg-ink text-white" : "bg-white text-zinc-600 ring-1 ring-line hover:bg-zinc-50"}`}
          >
            {FILTER_LABEL[f]}
          </button>
        ))}
      </div>

      {error && <div className="rounded-md bg-red-50 px-4 py-3 text-sm text-red-700">Cannot reach the API: {error}</div>}
      <div className="overflow-hidden rounded-lg border border-line bg-white">
        <table className="w-full text-left text-sm">
          <thead className="border-b border-line bg-zinc-50 text-xs uppercase tracking-wide text-zinc-500">
            <tr>
              <th className="px-4 py-2.5 font-medium">Date</th>
              <th className="px-4 py-2.5 font-medium">Patient</th>
              <th className="px-4 py-2.5 font-medium">Visit</th>
              <th className="px-4 py-2.5 font-medium">Primary code</th>
              <th className="px-4 py-2.5 font-medium">Status</th>
              <th className="px-4 py-2.5 text-right font-medium">Issues</th>
            </tr>
          </thead>
          <tbody>
            {visits === null && !error && (
              <tr>
                <td colSpan={6} className="px-4 py-8 text-center text-zinc-400">Loading…</td>
              </tr>
            )}
            {shown.map((v) => (
              <tr key={v.id} className="border-b border-line/70 last:border-0 hover:bg-zinc-50/80">
                <td className="whitespace-nowrap px-4 py-3 text-zinc-500">{dateLabel(v.visit_date)}</td>
                <td className="px-4 py-3">
                  <Link href={`/visits/${v.id}`} className="font-medium text-ink hover:underline">
                    {v.patient.name}
                  </Link>
                  <div className="text-xs text-zinc-400">
                    {age(v.patient.dob) !== null ? `${age(v.patient.dob)} y` : ""} {v.patient.sex !== "unknown" ? `· ${v.patient.sex}` : ""}
                  </div>
                </td>
                <td className="px-4 py-3">
                  <Link href={`/visits/${v.id}`} className="text-zinc-800 hover:underline">
                    {v.title}
                  </Link>
                  <div className="text-xs text-zinc-400">
                    {v.specialty_label} · {v.visit_type === "new" ? "new patient" : "established"} · {v.clinician}
                    {v.duration_s ? ` · ${clock(v.duration_s)}` : ""}
                  </div>
                </td>
                <td className="px-4 py-3">
                  {v.primary_code ? (
                    <span title={v.primary_code.description} className="rounded bg-zinc-100 px-1.5 py-0.5 font-mono text-xs text-zinc-700">
                      {v.primary_code.code}
                    </span>
                  ) : (
                    <span className="text-xs text-zinc-300">-</span>
                  )}
                </td>
                <td className="px-4 py-3">
                  <span className={`whitespace-nowrap rounded-full px-2 py-0.5 text-xs font-medium ${STATUS_STYLE[v.status]}`}>{STATUS_LABEL[v.status]}</span>
                </td>
                <td className="px-4 py-3 text-right tabular-nums">
                  {v.status === "drafted" ? (
                    v.open_issues ? <span className="font-medium text-red-600">{v.open_issues}</span> : <span className="text-emerald-600">0</span>
                  ) : (
                    <span className="text-zinc-300">-</span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="mt-3 flex items-center gap-1.5 text-xs text-zinc-400">
        <PlusIcon className="h-3 w-3" /> All 32 visits are synthetic dialogues spoken by two TTS voices; every name, date and number is fictional.
      </p>
      {open && <NewVisitDialog onClose={() => setOpen(false)} onCreated={load} />}
    </div>
  );
}
