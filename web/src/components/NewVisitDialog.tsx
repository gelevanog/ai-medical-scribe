"use client";

import { useRouter } from "next/navigation";
import { useEffect, useRef, useState } from "react";

import { MicIcon, UploadIcon } from "@/components/icons";
import { api } from "@/lib/api";
import { clock } from "@/lib/format";

type Props = { onClose: () => void; onCreated: () => void };

export function NewVisitDialog({ onClose, onCreated }: Props) {
  const router = useRouter();
  const [form, setForm] = useState({ name: "", dob: "", sex: "unknown", mrn: "", title: "", specialty: "primary_care", visit_type: "established" });
  const [audio, setAudio] = useState<{ blob: Blob; name: string } | null>(null);
  const [recording, setRecording] = useState(false);
  const [seconds, setSeconds] = useState(0);
  const [busy, setBusy] = useState("");
  const [error, setError] = useState("");
  const recorder = useRef<MediaRecorder | null>(null);
  const chunks = useRef<Blob[]>([]);

  useEffect(() => {
    if (!recording) return;
    const timer = setInterval(() => setSeconds((s) => s + 1), 1000);
    return () => clearInterval(timer);
  }, [recording]);

  async function startRecording() {
    setError("");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true } });
      const rec = new MediaRecorder(stream);
      chunks.current = [];
      rec.ondataavailable = (e) => e.data.size && chunks.current.push(e.data);
      rec.onstop = () => {
        stream.getTracks().forEach((t) => t.stop());
        setAudio({ blob: new Blob(chunks.current, { type: rec.mimeType || "audio/webm" }), name: "recording.webm" });
      };
      rec.start(1000);
      recorder.current = rec;
      setSeconds(0);
      setRecording(true);
    } catch (e) {
      setError(`Microphone not available: ${(e as Error).message}`);
    }
  }

  function stopRecording() {
    recorder.current?.stop();
    setRecording(false);
  }

  async function submit() {
    if (!form.name.trim()) {
      setError("Patient name is required (use a fictional one: this is a demo).");
      return;
    }
    setError("");
    try {
      setBusy("Creating the visit…");
      const visit = await api.create({
        title: form.title || "New visit",
        specialty: form.specialty,
        visit_type: form.visit_type,
        patient: { name: form.name, dob: form.dob, sex: form.sex, mrn: form.mrn },
      });
      if (audio) {
        setBusy("Uploading audio (stored encrypted)…");
        await api.upload(visit.id, audio.blob, audio.name);
        setBusy("Transcribing on this machine…");
        try {
          await api.transcribe(visit.id);
        } catch (e) {
          setError((e as Error).message);
        }
      }
      onCreated();
      router.push(`/visits/${visit.id}`);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy("");
    }
  }

  const field = "w-full rounded-md border border-line bg-white px-3 py-2 text-sm outline-none focus:border-brand focus:ring-2 focus:ring-brand/15";
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-ink/40 p-6" onClick={onClose}>
      <div className="w-full max-w-xl rounded-xl bg-white p-6 shadow-2xl" onClick={(e) => e.stopPropagation()}>
        <h2 className="text-lg font-semibold">New visit</h2>
        <p className="mt-1 text-xs text-zinc-500">
          Demo only: use fictional patients. Registration data is used to mask identifiers before any cloud model call.
        </p>
        <div className="mt-4 grid grid-cols-2 gap-3">
          <label className="col-span-2 text-xs font-medium text-zinc-600">
            Patient name
            <input className={field} value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })} placeholder="e.g. Rowan Ellery" />
          </label>
          <label className="text-xs font-medium text-zinc-600">
            Date of birth
            <input type="date" className={field} value={form.dob} onChange={(e) => setForm({ ...form, dob: e.target.value })} />
          </label>
          <label className="text-xs font-medium text-zinc-600">
            MRN
            <input className={field} value={form.mrn} onChange={(e) => setForm({ ...form, mrn: e.target.value })} />
          </label>
          <label className="text-xs font-medium text-zinc-600">
            Specialty
            <select className={field} value={form.specialty} onChange={(e) => setForm({ ...form, specialty: e.target.value })}>
              <option value="primary_care">Primary care</option>
              <option value="cardiology">Cardiology</option>
              <option value="dermatology">Dermatology</option>
              <option value="pediatrics">Pediatrics</option>
              <option value="mental_health">Mental health</option>
            </select>
          </label>
          <label className="text-xs font-medium text-zinc-600">
            Visit type
            <select className={field} value={form.visit_type} onChange={(e) => setForm({ ...form, visit_type: e.target.value })}>
              <option value="established">Established patient</option>
              <option value="new">New patient</option>
            </select>
          </label>
          <label className="col-span-2 text-xs font-medium text-zinc-600">
            Reason for visit
            <input className={field} value={form.title} onChange={(e) => setForm({ ...form, title: e.target.value })} placeholder="e.g. Cough for a week" />
          </label>
        </div>
        <div className="mt-4 rounded-lg border border-dashed border-line bg-zinc-50 p-4">
          <div className="flex items-center gap-3">
            {!recording ? (
              <button onClick={startRecording} className="flex items-center gap-2 rounded-md bg-red-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-red-700">
                <MicIcon className="h-4 w-4" /> Record
              </button>
            ) : (
              <button onClick={stopRecording} className="flex items-center gap-2 rounded-md bg-ink px-3 py-1.5 text-sm font-medium text-white">
                <span className="h-2 w-2 animate-pulse rounded-full bg-red-500" /> Stop {clock(seconds)}
              </button>
            )}
            <label className="flex cursor-pointer items-center gap-2 rounded-md bg-white px-3 py-1.5 text-sm ring-1 ring-line hover:bg-zinc-100">
              <UploadIcon className="h-4 w-4" /> Upload audio
              <input type="file" accept="audio/*" className="hidden" onChange={(e) => e.target.files?.[0] && setAudio({ blob: e.target.files[0], name: e.target.files[0].name })} />
            </label>
            <span className="text-xs text-zinc-500">{audio ? `${audio.name} · ${(audio.blob.size / 1e6).toFixed(1)} MB` : "Optional: add audio now or later"}</span>
          </div>
          <p className="mt-2 text-[11px] text-zinc-400">
            Transcription runs locally with faster-whisper (CPU) when the speech extra is installed. Tell the patient they are being recorded.
          </p>
        </div>
        {error && <div className="mt-3 rounded-md bg-red-50 px-3 py-2 text-xs text-red-700">{error}</div>}
        <div className="mt-5 flex justify-end gap-2">
          <button onClick={onClose} className="rounded-md px-3 py-2 text-sm text-zinc-600 hover:bg-zinc-100">Cancel</button>
          <button disabled={!!busy || recording} onClick={submit} className="rounded-md bg-brand px-4 py-2 text-sm font-medium text-white disabled:opacity-50">
            {busy || "Create visit"}
          </button>
        </div>
      </div>
    </div>
  );
}
