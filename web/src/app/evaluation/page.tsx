"use client";

import { useEffect, useState } from "react";

import { api } from "@/lib/api";

type Run = Record<string, unknown> & { by_specialty?: Record<string, Record<string, number | null>> };
type Summary = {
  available: boolean;
  runs?: Record<string, Run>;
  asr?: Record<string, Record<string, unknown>>;
  canary?: Record<string, unknown>;
  guards?: { catch_rate?: Record<string, { trials: number; caught: number; rate: number }> };
  phi_challenge_frozen?: Record<string, { recall: number; gold_values: number; covered: number } | null>;
  phi?: Record<string, { recall: number; gold_values: number; covered: number }>;
};

const pct = (v: unknown) => (typeof v === "number" ? `${(100 * v).toFixed(1)}%` : "-");
const ROWS: [string, string, (r: Run) => string][] = [
  ["Fact recall", "gold SOAP facts found with the right polarity", (r) => pct(r.fact_recall)],
  ["... in the right section", "", (r) => pct(r.fact_recall_section_correct)],
  ["Statements matching gold", "lower bound on precision", (r) => pct(r.gold_matched_precision)],
  ["Unsupported statements", "no transcript evidence (grounding guard)", (r) => pct(r.unsupported_rate)],
  ["Negation errors", "of the pertinent negatives in gold", (r) => `${r.negation_errors} / ${r.negatives_in_gold}`],
  ["Trap errors", "family history, hypotheticals, corrections", (r) => String(r.trap_errors)],
  ["Medication recall", "", (r) => pct(r.medication_recall)],
  ["ICD-10-CM primary, top-1", "", (r) => pct(r.icd_top1)],
  ["ICD-10-CM primary, top-3", "", (r) => pct(r.icd_top3)],
  ["E/M exact / within one level", "", (r) => `${pct(r.em_exact)} / ${pct(r.em_within_one)}`],
  ["PHI leaks to the model", "identifiers found in outgoing payloads", (r) => `${r.phi_leaks} in ${r.payloads_checked}`],
];

export default function EvaluationPage() {
  const [data, setData] = useState<Summary | null>(null);
  const [error, setError] = useState("");
  useEffect(() => {
    api.evaluation().then((d) => setData(d as Summary)).catch((e: Error) => setError(e.message));
  }, []);
  const runs = Object.entries(data?.runs ?? {});
  return (
    <div className="mx-auto max-w-6xl px-8 py-8">
      <h1 className="text-2xl font-semibold tracking-tight">Evaluation</h1>
      <p className="mt-1 text-sm text-zinc-500">
        32 synthetic visits (5 specialties) with gold SOAP facts, medications, codes, identifiers and traps. Numbers come from
        the committed artifacts in <span className="font-mono">results/</span>.
      </p>
      {error && <div className="mt-4 rounded-md bg-red-50 px-4 py-3 text-sm text-red-700">{error}</div>}
      {data && !data.available && <div className="mt-4 text-sm text-zinc-500">No results yet: run <span className="font-mono">notewell eval report</span>.</div>}
      {runs.length > 0 && (
        <div className="mt-6 overflow-x-auto rounded-lg border border-line bg-white">
          <table className="w-full text-left text-[13px]">
            <thead className="border-b border-line bg-zinc-50 text-xs text-zinc-500">
              <tr>
                <th className="px-4 py-2 font-medium">Metric</th>
                {runs.map(([name, r]) => (
                  <th key={name} className="px-4 py-2 font-medium">
                    <div className="text-zinc-800">{name}</div>
                    <div className="font-mono text-[10.5px] font-normal">{String(r.model)}</div>
                    <div className="font-normal">{String(r.transcripts)} transcripts</div>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {ROWS.map(([label, hint, fn]) => (
                <tr key={label} className="border-b border-line/70 last:border-0">
                  <td className="px-4 py-2">
                    <div className="font-medium text-zinc-800">{label}</div>
                    {hint && <div className="text-[11px] text-zinc-400">{hint}</div>}
                  </td>
                  {runs.map(([name, r]) => (
                    <td key={name} className="px-4 py-2 tabular-nums">{fn(r)}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      <div className="mt-6 grid grid-cols-3 gap-4">
        <div className="rounded-lg border border-line bg-white p-4">
          <h2 className="text-sm font-semibold">Speech recognition (CPU)</h2>
          <ul className="mt-2 space-y-1 text-[13px]">
            {Object.entries(data?.asr ?? {}).map(([name, a]) => (
              <li key={name} className="flex justify-between gap-2">
                <span className="font-mono text-xs">{name}</span>
                <span className="tabular-nums">WER {pct(a.wer)} · roles {pct(a.role_accuracy)}</span>
              </li>
            ))}
          </ul>
        </div>
        <div className="rounded-lg border border-line bg-white p-4">
          <h2 className="text-sm font-semibold">PHI masking</h2>
          <ul className="mt-2 space-y-1 text-[13px]">
            {Object.entries(data?.phi_challenge_frozen ?? {}).map(([name, v]) =>
              v ? (
                <li key={name} className="flex justify-between"><span>held-out {name.replaceAll("_", " ")}</span><span className="tabular-nums">{pct(v.recall)}</span></li>
              ) : null,
            )}
            {data?.canary && (
              <li className="flex justify-between"><span>canary leaks</span><span className="tabular-nums">{String(data.canary.leaks)} in {String(data.canary.payloads_checked)} payloads</span></li>
            )}
          </ul>
        </div>
        <div className="rounded-lg border border-line bg-white p-4">
          <h2 className="text-sm font-semibold">Guards on corrupted notes</h2>
          <ul className="mt-2 space-y-1 text-[13px]">
            {Object.entries(data?.guards?.catch_rate ?? {}).map(([k, v]) => (
              <li key={k} className="flex justify-between"><span>{k.replaceAll("_", " ")}</span><span className="tabular-nums">{v.caught}/{v.trials}</span></li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  );
}
