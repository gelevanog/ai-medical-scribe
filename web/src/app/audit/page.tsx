"use client";

import { useEffect, useState } from "react";

import { api, type AuditRow } from "@/lib/api";

export default function AuditPage() {
  const [rows, setRows] = useState<AuditRow[] | null>(null);
  const [error, setError] = useState("");
  useEffect(() => {
    api.audit().then(setRows).catch((e: Error) => setError(e.message));
  }, []);
  return (
    <div className="mx-auto max-w-6xl px-8 py-8">
      <h1 className="text-2xl font-semibold tracking-tight">Audit log</h1>
      <p className="mt-1 text-sm text-zinc-500">
        Who did what to which visit, and when. Entries hold ids, counts, model names and hashes, never names, transcripts or note
        text.
      </p>
      {error && <div className="mt-4 rounded-md bg-red-50 px-4 py-3 text-sm text-red-700">{error}</div>}
      <div className="mt-5 overflow-hidden rounded-lg border border-line bg-white">
        <table className="w-full text-left text-[13px]">
          <thead className="border-b border-line bg-zinc-50 text-xs uppercase tracking-wide text-zinc-500">
            <tr>
              <th className="px-4 py-2 font-medium">Time (UTC)</th>
              <th className="px-4 py-2 font-medium">Actor</th>
              <th className="px-4 py-2 font-medium">Action</th>
              <th className="px-4 py-2 font-medium">Visit</th>
              <th className="px-4 py-2 font-medium">Detail</th>
            </tr>
          </thead>
          <tbody>
            {(rows ?? []).map((r, i) => (
              <tr key={i} className="border-b border-line/70 align-top last:border-0">
                <td className="whitespace-nowrap px-4 py-2 font-mono text-xs text-zinc-500">{r.ts.replace("T", " ").replace("+00:00", "")}</td>
                <td className="px-4 py-2">{r.actor}</td>
                <td className="px-4 py-2 font-mono text-xs">{r.action}</td>
                <td className="px-4 py-2 font-mono text-xs">{r.visit_id ?? "-"}</td>
                <td className="px-4 py-2 font-mono text-[11px] text-zinc-500">{JSON.stringify(r.detail)}</td>
              </tr>
            ))}
            {rows && rows.length === 0 && (
              <tr>
                <td colSpan={5} className="px-4 py-6 text-center text-zinc-400">No entries yet.</td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
