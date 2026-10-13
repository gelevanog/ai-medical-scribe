"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

import { AuditIcon, EvalIcon, ListIcon, LogoMark, ShieldIcon } from "@/components/icons";

const NAV = [
  { href: "/", label: "Visits", icon: ListIcon },
  { href: "/evaluation", label: "Evaluation", icon: EvalIcon },
  { href: "/audit", label: "Audit log", icon: AuditIcon },
  { href: "/privacy", label: "Privacy & safety", icon: ShieldIcon },
];

export function Sidebar() {
  const path = usePathname();
  return (
    <aside className="sticky top-0 flex h-screen w-56 shrink-0 flex-col bg-ink text-zinc-300">
      <Link href="/" className="flex items-center gap-2.5 px-5 pb-6 pt-5 text-white">
        <LogoMark className="h-8 w-8 text-white" />
        <div>
          <div className="text-[17px] font-semibold tracking-tight">Notewell</div>
          <div className="text-[11px] text-zinc-400">AI medical scribe</div>
        </div>
      </Link>
      <nav className="flex flex-col gap-0.5 px-3">
        {NAV.map(({ href, label, icon: Icon }) => {
          const active = href === "/" ? path === "/" || path.startsWith("/visits") : path.startsWith(href);
          return (
            <Link
              key={href}
              href={href}
              className={`flex items-center gap-3 rounded-md px-3 py-2 text-[13.5px] ${active ? "bg-white/10 text-white" : "hover:bg-white/5 hover:text-white"}`}
            >
              <Icon className="h-4 w-4" />
              {label}
            </Link>
          );
        })}
      </nav>
      <div className="mt-auto space-y-2 px-5 pb-5 text-[11px] leading-snug text-zinc-400">
        <div className="rounded-md border border-amber-400/30 bg-amber-400/10 px-2.5 py-2 text-amber-200">
          Demo with synthetic patients. Not a medical device, not for clinical use.
        </div>
        <div>Signed in as Dr. Demo (clinician)</div>
      </div>
    </aside>
  );
}
