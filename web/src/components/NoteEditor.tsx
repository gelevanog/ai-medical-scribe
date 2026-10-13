"use client";

import { useState } from "react";

import { AlertIcon, CheckIcon, EditIcon, LockIcon, PlusIcon, TrashIcon } from "@/components/icons";
import { type Flag, type Note, type Section, SECTIONS, type Statement } from "@/lib/api";
import { FLAG_LABEL, SECTION_LABEL } from "@/lib/format";

type Actions = {
  edit: (sid: string, text: string) => Promise<void>;
  remove: (sid: string) => Promise<void>;
  add: (section: Section, text: string) => Promise<void>;
  acknowledge: (target: string, kind: string) => Promise<void>;
  accept: (section: Section) => Promise<void>;
  reopen: (section: Section) => Promise<void>;
};

type Props = {
  note: Note;
  locked: boolean;
  actions: Actions;
  onHover: (turns: number[]) => void;
  focused: string | null;
};

const SUPPORT_DOT: Record<string, string> = {
  supported: "bg-emerald-500",
  partial: "bg-amber-500",
  unsupported: "bg-red-500",
};

function FlagChip({ flag, onAck, locked }: { flag: Flag; onAck: () => void; locked: boolean }) {
  const tone =
    flag.acknowledged || flag.severity === "info"
      ? "bg-zinc-100 text-zinc-500"
      : flag.severity === "error"
        ? "bg-red-50 text-red-700 ring-1 ring-red-200"
        : "bg-amber-50 text-amber-800 ring-1 ring-amber-200";
  return (
    <div className={`mt-1 flex items-start gap-1.5 rounded px-2 py-1 text-[11.5px] ${tone}`}>
      <AlertIcon className="mt-px h-3.5 w-3.5 shrink-0" />
      <div className="min-w-0 flex-1">
        <span className="font-semibold">{FLAG_LABEL[flag.kind] ?? flag.kind}:</span> {flag.message}
      </div>
      {!flag.acknowledged && flag.severity !== "info" && !locked && (
        <button onClick={onAck} className="shrink-0 rounded px-1.5 font-medium underline-offset-2 hover:underline">
          Accept risk
        </button>
      )}
      {flag.acknowledged && <span className="shrink-0 text-[10.5px]">acknowledged</span>}
    </div>
  );
}

function StatementRow({ s, locked, actions, onHover, focused }: { s: Statement; locked: boolean; actions: Actions; onHover: Props["onHover"]; focused: boolean }) {
  const [editing, setEditing] = useState(false);
  const [text, setText] = useState(s.text);
  const turns = s.evidence.filter((e) => e.verified).map((e) => e.turn);
  const quotes = s.evidence.filter((e) => e.verified && e.quote).map((e) => `#${e.turn}: "${e.quote}"`);
  return (
    <li
      onMouseEnter={() => onHover(turns)}
      onMouseLeave={() => onHover([])}
      className={`group relative rounded-md px-2 py-1.5 ${focused ? "bg-yellow-50 ring-1 ring-yellow-300" : "hover:bg-zinc-50"}`}
    >
      <div className="flex items-start gap-2">
        <span className={`mt-[7px] h-2 w-2 shrink-0 rounded-full ${SUPPORT_DOT[s.support]}`} title={s.support} />
        {editing ? (
          <div className="flex-1">
            <textarea
              value={text}
              onChange={(e) => setText(e.target.value)}
              rows={2}
              className="w-full rounded border border-brand/40 px-2 py-1 text-[13.5px] outline-none ring-2 ring-brand/10"
              autoFocus
            />
            <div className="mt-1 flex gap-2 text-xs">
              <button
                className="rounded bg-brand px-2 py-0.5 font-medium text-white"
                onClick={async () => {
                  await actions.edit(s.id, text);
                  setEditing(false);
                }}
              >
                Save
              </button>
              <button className="text-zinc-500" onClick={() => setEditing(false)}>
                Cancel
              </button>
            </div>
          </div>
        ) : (
          <div className="min-w-0 flex-1 text-[13.5px] leading-snug text-zinc-800">
            <span title={quotes.join("\n") || "no verified quote"}>{s.text}</span>
            {s.origin === "clinician" && <span className="ml-1.5 rounded bg-sky-50 px-1 text-[10px] font-medium text-sky-700">added</span>}
            {s.edited && s.origin === "model" && <span className="ml-1.5 rounded bg-zinc-100 px-1 text-[10px] text-zinc-500">edited</span>}
            {turns.length > 0 && <span className="ml-1.5 font-mono text-[10.5px] text-zinc-400">{turns.map((t) => `#${t}`).join(" ")}</span>}
          </div>
        )}
        {!locked && !editing && (
          <div className="flex shrink-0 gap-0.5 opacity-0 transition group-hover:opacity-100">
            <button title="Edit" onClick={() => setEditing(true)} className="rounded p-1 text-zinc-400 hover:bg-zinc-100 hover:text-zinc-700">
              <EditIcon className="h-3.5 w-3.5" />
            </button>
            <button title="Remove" onClick={() => actions.remove(s.id)} className="rounded p-1 text-zinc-400 hover:bg-red-50 hover:text-red-600">
              <TrashIcon className="h-3.5 w-3.5" />
            </button>
          </div>
        )}
      </div>
      {s.flags.length > 0 && (
        <div className="ml-4">
          {s.flags.map((f, i) => (
            <FlagChip key={i} flag={f} locked={locked} onAck={() => actions.acknowledge(s.id, f.kind)} />
          ))}
        </div>
      )}
    </li>
  );
}

function AddStatement({ section, add }: { section: Section; add: Actions["add"] }) {
  const [open, setOpen] = useState(false);
  const [text, setText] = useState("");
  if (!open)
    return (
      <button onClick={() => setOpen(true)} className="mt-1 flex items-center gap-1 px-2 text-xs text-zinc-400 hover:text-brand">
        <PlusIcon className="h-3 w-3" /> Add statement
      </button>
    );
  return (
    <div className="mt-1 flex gap-2 px-2">
      <input
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="Type a statement; it is checked against the transcript"
        className="flex-1 rounded border border-line px-2 py-1 text-sm outline-none focus:border-brand"
        autoFocus
        onKeyDown={async (e) => {
          if (e.key === "Enter" && text.trim()) {
            await add(section, text);
            setText("");
            setOpen(false);
          }
        }}
      />
      <button onClick={() => setOpen(false)} className="text-xs text-zinc-500">
        Cancel
      </button>
    </div>
  );
}

export function NoteEditor({ note, locked, actions, onHover, focused }: Props) {
  return (
    <div className="space-y-3">
      {SECTIONS.map((section) => {
        const items = note.statements.filter((s) => s.section === section);
        const accepted = note.section_status[section] === "accepted";
        const open = items.flatMap((s) => s.flags).filter((f) => f.severity === "error" && !f.acknowledged).length;
        return (
          <section key={section} className={`rounded-lg border bg-white ${accepted ? "border-emerald-200" : "border-line"}`}>
            <header className="flex items-center justify-between border-b border-line/70 px-3 py-2">
              <div className="flex items-center gap-2">
                <span className="flex h-6 w-6 items-center justify-center rounded bg-ink text-xs font-bold text-white">{section[0].toUpperCase()}</span>
                <h3 className="text-sm font-semibold">{SECTION_LABEL[section]}</h3>
                <span className="text-xs text-zinc-400">{items.length} statements</span>
                {open > 0 && <span className="rounded-full bg-red-50 px-2 text-[11px] font-medium text-red-700">{open} to resolve</span>}
              </div>
              {locked ? (
                <LockIcon className="h-4 w-4 text-zinc-400" />
              ) : accepted ? (
                <button onClick={() => actions.reopen(section)} className="flex items-center gap-1 rounded-full bg-emerald-50 px-2.5 py-0.5 text-xs font-medium text-emerald-700 hover:bg-emerald-100">
                  <CheckIcon className="h-3.5 w-3.5" /> Accepted
                </button>
              ) : (
                <button
                  onClick={() => actions.accept(section)}
                  disabled={open > 0}
                  title={open > 0 ? "Resolve or acknowledge the flagged statements first" : "Accept this section"}
                  className="rounded-full px-2.5 py-0.5 text-xs font-medium text-brand ring-1 ring-brand/40 hover:bg-brand-soft disabled:cursor-not-allowed disabled:opacity-40"
                >
                  Accept section
                </button>
              )}
            </header>
            <ul className="space-y-0.5 px-1.5 py-1.5">
              {items.map((s) => (
                <StatementRow key={s.id} s={s} locked={locked} actions={actions} onHover={onHover} focused={focused === s.id} />
              ))}
            </ul>
            {section === "subjective" && (note.medications.length > 0 || note.allergies.length > 0) && (
              <div className="border-t border-line/70 px-3 py-2">
                <div className="text-[11px] font-semibold uppercase tracking-wide text-zinc-500">Medications</div>
                <ul className="mt-1 space-y-0.5">
                  {note.medications.map((m, i) => (
                    <li
                      key={i}
                      onMouseEnter={() => onHover(m.evidence.filter((e) => e.verified).map((e) => e.turn))}
                      onMouseLeave={() => onHover([])}
                      className="rounded px-1 text-[13px] hover:bg-zinc-50"
                    >
                      <span className="font-medium">{m.name}</span> <span className="text-zinc-600">{[m.dose, m.frequency, m.route].filter(Boolean).join(", ")}</span>
                      <span className="ml-1.5 rounded bg-zinc-100 px-1 text-[10.5px] text-zinc-600">{m.status}</span>
                      {m.flags.map((f, j) => (
                        <FlagChip key={j} flag={f} locked={locked} onAck={() => actions.acknowledge(`med:${i}`, f.kind)} />
                      ))}
                    </li>
                  ))}
                </ul>
                <div className="mt-2 text-[11px] font-semibold uppercase tracking-wide text-zinc-500">Allergies</div>
                <div className="text-[13px] text-zinc-700">
                  {note.allergies.length ? note.allergies.map((a) => `${a.substance}${a.reaction ? ` (${a.reaction})` : ""}`).join("; ") : "No known drug allergies"}
                </div>
              </div>
            )}
            {!locked && <div className="pb-2"><AddStatement section={section} add={actions.add} /></div>}
          </section>
        );
      })}
    </div>
  );
}
