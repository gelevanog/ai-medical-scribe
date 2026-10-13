export function age(dob: string, on = new Date()): number | null {
  const m = /^(\d{4})-(\d{2})-(\d{2})/.exec(dob);
  if (!m) return null;
  const birth = new Date(Number(m[1]), Number(m[2]) - 1, Number(m[3]));
  let years = on.getFullYear() - birth.getFullYear();
  const before = on.getMonth() < birth.getMonth() || (on.getMonth() === birth.getMonth() && on.getDate() < birth.getDate());
  if (before) years -= 1;
  return years;
}

export function clock(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined || Number.isNaN(seconds)) return "";
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

export function dateLabel(iso: string): string {
  const d = new Date(`${iso}T12:00:00`);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" });
}

export const STATUS_STYLE: Record<string, string> = {
  new: "bg-zinc-100 text-zinc-600",
  transcribed: "bg-sky-50 text-sky-700",
  drafted: "bg-amber-50 text-amber-800",
  signed: "bg-emerald-50 text-emerald-700",
  exported: "bg-teal-50 text-teal-800",
};

export const STATUS_LABEL: Record<string, string> = {
  new: "New",
  transcribed: "Transcribed",
  drafted: "Draft ready",
  signed: "Signed",
  exported: "Exported to EHR",
};

export const SPEAKER_STYLE: Record<string, { label: string; chip: string; text: string }> = {
  clinician: { label: "Clinician", chip: "bg-clinician-soft text-clinician", text: "text-clinician" },
  patient: { label: "Patient", chip: "bg-patient-soft text-patient", text: "text-patient" },
  caregiver: { label: "Caregiver", chip: "bg-caregiver-soft text-caregiver", text: "text-caregiver" },
  unknown: { label: "Speaker", chip: "bg-zinc-100 text-zinc-600", text: "text-zinc-600" },
};

export const SECTION_LABEL: Record<string, string> = {
  subjective: "Subjective",
  objective: "Objective",
  assessment: "Assessment",
  plan: "Plan",
};

export const FLAG_LABEL: Record<string, string> = {
  unsupported: "Not in transcript",
  partial_support: "Partly supported",
  negation: "Negation conflict",
  attribution: "Someone else's",
  hypothetical: "Hypothetical",
  dose_range: "Dose range",
  dose_unit: "Dose unit",
  dose_mismatch: "Dose check",
  unknown_drug: "Unknown drug",
  allergy_conflict: "Allergy conflict",
  code_invalid: "Invalid code",
  code_not_billable: "Not billable",
  code_unsupported: "Weak evidence",
  code_negated: "Denied finding",
  em_mismatch: "Level check",
  em_visit_type: "Wrong code family",
  placeholder: "Placeholder",
  readability: "Readability",
};
