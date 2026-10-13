// Types and calls for the Notewell API. The browser talks to the API directly (CORS); its URL is inlined at build time.

export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export type Section = "subjective" | "objective" | "assessment" | "plan";
export const SECTIONS: Section[] = ["subjective", "objective", "assessment", "plan"];

export type Flag = {
  kind: string;
  severity: "error" | "warning" | "info";
  message: string;
  acknowledged: boolean;
  detail: Record<string, unknown>;
};

export type Evidence = { turn: number; quote: string; verified: boolean; score: number };

export type Statement = {
  id: string;
  section: Section;
  text: string;
  evidence: Evidence[];
  support: "supported" | "partial" | "unsupported";
  flags: Flag[];
  origin: "model" | "clinician";
  edited: boolean;
};

export type Medication = {
  name: string;
  dose: string;
  frequency: string;
  route: string;
  status: string;
  evidence: Evidence[];
  flags: Flag[];
};

export type Allergy = { substance: string; reaction: string; evidence: Evidence[] };

export type Code = {
  system: "ICD-10-CM" | "CPT";
  code: string;
  description: string;
  rank: number;
  rationale: string;
  evidence: Evidence[];
  valid: boolean;
  billable: boolean;
  flags: Flag[];
  accepted: boolean | null;
};

export type MDM = { problems: string; data: string; risk: string; level: string; code: string; reasons: string[] };

export type Note = {
  visit_id: string;
  statements: Statement[];
  medications: Medication[];
  allergies: Allergy[];
  after_visit_summary: string;
  avs_flags: Flag[];
  icd10: Code[];
  em: Code | null;
  procedures: Code[];
  mdm: MDM | null;
  section_status: Record<Section, "draft" | "accepted">;
  model: string;
  created_at: string | null;
  signed_by: string | null;
  signed_at: string | null;
  timings: Record<string, number>;
  usage: Record<string, number>;
  phi: Record<string, number>;
};

export type Turn = { idx: number; speaker: "clinician" | "patient" | "caregiver" | "unknown"; text: string; start: number | null; end: number | null };

export type Transcript = {
  turns: Turn[];
  source: "gold" | "asr";
  asr_model: string;
  duration_s: number;
  timings: Record<string, number>;
  diarization: Record<string, unknown>;
};

export type Patient = { name: string; dob: string; sex: string; mrn: string; phone: string; address: string; email: string };

export type VisitSummary = {
  id: string;
  title: string;
  specialty: string;
  specialty_label: string;
  visit_type: "new" | "established";
  visit_date: string;
  status: "new" | "transcribed" | "drafted" | "signed" | "exported";
  patient: { name: string; dob: string; sex: string };
  clinician: string;
  has_audio: boolean;
  duration_s: number | null;
  open_issues: number;
  primary_code: { code: string; description: string } | null;
  model: string | null;
  synthetic: boolean;
};

export type Visit = {
  id: string;
  title: string;
  specialty: string;
  visit_type: "new" | "established";
  visit_date: string;
  patient: Patient;
  clinician: { name: string; role: string };
  status: VisitSummary["status"];
  audio: string | null;
  transcript: Transcript | null;
  note: Note | null;
  synthetic: boolean;
  summary: VisitSummary;
};

export type AuditRow = { ts: string; actor: string; action: string; visit_id: string | null; detail: Record<string, unknown> };

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_URL}${path}`, {
    ...init,
    headers: init?.body && !(init.body instanceof FormData) ? { "Content-Type": "application/json", ...init?.headers } : init?.headers,
    cache: "no-store",
  });
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      // keep the status line
    }
    throw new Error(detail);
  }
  return (await response.json()) as T;
}

const post = <T,>(path: string, body?: unknown) => call<T>(path, { method: "POST", body: body === undefined ? undefined : JSON.stringify(body) });

export const api = {
  health: () => call<Record<string, unknown>>("/health"),
  visits: () => call<VisitSummary[]>("/api/visits"),
  visit: (id: string) => call<Visit>(`/api/visits/${id}`),
  masked: (id: string) => call<{ turns: Turn[]; counts: Record<string, number> }>(`/api/visits/${id}/masked`),
  create: (body: { title: string; specialty: string; visit_type: string; patient: Partial<Patient> & { name: string } }) =>
    post<VisitSummary>("/api/visits", body),
  upload: (id: string, blob: Blob, filename: string) => {
    const form = new FormData();
    form.append("file", blob, filename);
    return call<VisitSummary>(`/api/visits/${id}/audio`, { method: "POST", body: form });
  },
  transcribe: (id: string) => post<Visit>(`/api/visits/${id}/transcribe`),
  draft: (id: string) => post<Visit>(`/api/visits/${id}/draft`),
  edit: (id: string, sid: string, text: string) => call<Visit>(`/api/visits/${id}/statements/${sid}`, { method: "PUT", body: JSON.stringify({ text }) }),
  remove: (id: string, sid: string) => call<Visit>(`/api/visits/${id}/statements/${sid}`, { method: "DELETE" }),
  add: (id: string, section: Section, text: string) => post<Visit>(`/api/visits/${id}/statements`, { section, text }),
  acknowledge: (id: string, target: string, kind: string) => post<Visit>(`/api/visits/${id}/acknowledge`, { target, kind }),
  accept: (id: string, section: Section) => post<Visit>(`/api/visits/${id}/sections/${section}/accept`),
  reopen: (id: string, section: Section) => post<Visit>(`/api/visits/${id}/sections/${section}/reopen`),
  code: (id: string, code: string, accepted: boolean | null, system = "any") => post<Visit>(`/api/visits/${id}/codes`, { code, accepted, system }),
  addCode: (id: string, code: string) => post<Visit>(`/api/visits/${id}/codes/add`, { code }),
  searchCodes: (q: string) => call<{ code: string; title: string; score: number }[]>(`/api/codes/search?q=${encodeURIComponent(q)}`),
  sign: (id: string) => post<Visit>(`/api/visits/${id}/sign`, { attestation: true }),
  exportFhir: (id: string) => post<{ server: string; resources: string[]; bundle_entries: number }>(`/api/visits/${id}/export/fhir`),
  fhir: (id: string) => call<Record<string, unknown>>(`/api/visits/${id}/fhir`),
  audit: (visitId?: string) => call<AuditRow[]>(`/api/audit${visitId ? `?visit_id=${visitId}` : ""}`),
  evaluation: () => call<Record<string, unknown>>("/api/eval"),
};

export const audioUrl = (id: string) => `${API_URL}/api/visits/${id}/audio`;
export const pdfUrl = (id: string) => `${API_URL}/api/visits/${id}/note.pdf`;
