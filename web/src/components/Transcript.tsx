"use client";

import { useEffect, useRef } from "react";

import type { Turn } from "@/lib/api";
import { clock, SPEAKER_STYLE } from "@/lib/format";

type Props = {
  turns: Turn[];
  highlight: Set<number>;
  masked: boolean;
  onSeek: (seconds: number) => void;
  playing: number | null;
};

function renderMasked(text: string) {
  const parts = text.split(/(\[(?:PATIENT|DOB|AGE_90_PLUS|[A-Z]+_\d+)\])/g);
  return parts.map((p, i) =>
    /^\[(?:PATIENT|DOB|AGE_90_PLUS|[A-Z]+_\d+)\]$/.test(p) ? (
      <mark key={i} className="phi">
        {p.slice(1, -1)}
      </mark>
    ) : (
      <span key={i}>{p}</span>
    ),
  );
}

export function Transcript({ turns, highlight, masked, onSeek, playing }: Props) {
  const container = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!highlight.size || !container.current) return;
    const first = Math.min(...highlight);
    const el = container.current.querySelector<HTMLElement>(`[data-turn="${first}"]`);
    if (el) {
      const box = container.current;
      const top = el.offsetTop - box.offsetTop - box.clientHeight / 3;
      box.scrollTo({ top, behavior: "smooth" });
    }
  }, [highlight]);
  return (
    <div ref={container} className="scroll-thin h-full space-y-1 overflow-y-auto pr-1">
      {turns.map((t) => {
        const style = SPEAKER_STYLE[t.speaker] ?? SPEAKER_STYLE.unknown;
        const hit = highlight.has(t.idx);
        const live = playing !== null && t.start !== null && t.end !== null && playing >= t.start && playing <= t.end + 0.4;
        return (
          <div
            key={t.idx}
            data-turn={t.idx}
            className={`group rounded-md px-2.5 py-1.5 text-[13px] leading-relaxed ${hit ? "evidence-hit" : live ? "bg-sky-50" : "hover:bg-zinc-50"}`}
          >
            <div className="mb-0.5 flex items-center gap-2 text-[11px]">
              <span className={`rounded px-1.5 py-px font-semibold ${style.chip}`}>{style.label}</span>
              {t.start !== null && (
                <button onClick={() => onSeek(t.start ?? 0)} className="font-mono text-zinc-400 hover:text-brand" title="Play from here">
                  {clock(t.start)}
                </button>
              )}
              <span className="font-mono text-zinc-300">#{t.idx}</span>
            </div>
            <div className="text-zinc-800">{masked ? renderMasked(t.text) : t.text}</div>
          </div>
        );
      })}
    </div>
  );
}
