const ITEMS: [string, string][] = [
  ["What leaves this machine", "Only the masked transcript (placeholders such as [PATIENT], [NAME_2], [PHONE_1]) and the masked note draft, sent to the configured LLM. Audio never leaves: speech recognition runs locally on CPU."],
  ["How identifiers are masked", "Registered demographics (name, date of birth, phone, MRN, address, email) are matched exactly and fuzzily, then patterns (phones, e-mails, IDs, dates, addresses, ages over 89), name gazetteers and context cues find the rest. An egress check scans every outgoing request again and blocks it if a known identifier remains."],
  ["Restoring", "The model's answer is restored locally from an encrypted per-visit vault. Clinicians see real names; the model never does."],
  ["Grounding", "Every statement is linked to a transcript quote. Statements without support, numbers or drugs nobody said, negation conflicts (\"denies chest pain\" vs \"chest pain\"), family history attributed to the patient and hypotheticals are flagged."],
  ["Human in the loop", "Each section must be accepted and every error-level flag resolved or explicitly acknowledged before the clinician can sign. Only signed notes can be exported (FHIR R4, PDF)."],
  ["Encryption at rest", "Visit documents, vaults and uploaded audio are AES-256-GCM encrypted in SQLite, bound to their record id. In production the key belongs in a KMS/HSM."],
  ["Audit log", "Every view of the masked data, draft, edit, acceptance, signature and export is logged with actor and time; never with identifiers or clinical text."],
  ["Retention", "`notewell purge` deletes audio and vaults of signed visits after the retention period (NOTEWELL_RETENTION_DAYS, default 30)."],
  ["Access control", "Bearer tokens with roles (NOTEWELL_AUTH_TOKENS); only clinicians may edit, sign or export. The demo runs as a single demo clinician."],
  ["Before real patient data", "Sign a Business Associate Agreement with every vendor that touches PHI (LLM provider, hosting), use a BAA-covered model endpoint (free OpenRouter models are not one), run a risk assessment, add SSO/MFA, backups and monitoring. This demo is not HIPAA compliant and makes no such claim."],
];

export default function PrivacyPage() {
  return (
    <div className="mx-auto max-w-4xl px-8 py-8">
      <h1 className="text-2xl font-semibold tracking-tight">Privacy and safety</h1>
      <div className="mt-3 rounded-lg border border-amber-300 bg-amber-50 px-4 py-3 text-sm text-amber-900">
        Notewell is a portfolio demo with 100% synthetic data. It is not a medical device, it is not cleared or approved for clinical
        use, and it must not be used with real patient information.
      </div>
      <dl className="mt-6 divide-y divide-line rounded-lg border border-line bg-white">
        {ITEMS.map(([title, text]) => (
          <div key={title} className="grid grid-cols-[200px_1fr] gap-4 px-5 py-3.5">
            <dt className="text-sm font-semibold text-zinc-800">{title}</dt>
            <dd className="text-sm leading-relaxed text-zinc-600">{text}</dd>
          </div>
        ))}
      </dl>
    </div>
  );
}
