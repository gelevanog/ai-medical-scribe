import type { SVGProps } from "react";

type P = SVGProps<SVGSVGElement>;
const base = { fill: "none", stroke: "currentColor", strokeWidth: 1.8, strokeLinecap: "round", strokeLinejoin: "round" } as const;

export const LogoMark = (p: P) => (
  <svg viewBox="0 0 32 32" {...p}>
    <path d="M9 7h11l4 4v14H9z" fill="none" stroke="currentColor" strokeWidth="2" strokeLinejoin="round" />
    <path d="M12 15h3l1.5-3 2 6 1.5-3H22" fill="none" stroke="#2dd4bf" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
  </svg>
);
export const ListIcon = (p: P) => (
  <svg viewBox="0 0 24 24" {...base} {...p}><path d="M8 6h12M8 12h12M8 18h12M4 6h.01M4 12h.01M4 18h.01" /></svg>
);
export const EvalIcon = (p: P) => (
  <svg viewBox="0 0 24 24" {...base} {...p}><path d="M4 20V10M10 20V4M16 20v-7M22 20H2" /></svg>
);
export const AuditIcon = (p: P) => (
  <svg viewBox="0 0 24 24" {...base} {...p}><path d="M6 3h9l4 4v14H6z" /><path d="M9 11h7M9 15h7M9 7h3" /></svg>
);
export const ShieldIcon = (p: P) => (
  <svg viewBox="0 0 24 24" {...base} {...p}><path d="M12 3l8 3v6c0 4.5-3.3 8-8 9-4.7-1-8-4.5-8-9V6z" /><path d="M9 12l2 2 4-4" /></svg>
);
export const MicIcon = (p: P) => (
  <svg viewBox="0 0 24 24" {...base} {...p}><rect x="9" y="3" width="6" height="11" rx="3" /><path d="M5 11a7 7 0 0 0 14 0M12 18v3" /></svg>
);
export const SparkIcon = (p: P) => (
  <svg viewBox="0 0 24 24" {...base} {...p}><path d="M12 3v4M12 17v4M3 12h4M17 12h4M6 6l2.5 2.5M15.5 15.5L18 18M6 18l2.5-2.5M15.5 8.5L18 6" /></svg>
);
export const CheckIcon = (p: P) => (
  <svg viewBox="0 0 24 24" {...base} {...p}><path d="M5 12.5l4.5 4.5L19 7.5" /></svg>
);
export const AlertIcon = (p: P) => (
  <svg viewBox="0 0 24 24" {...base} {...p}><path d="M12 4l9 16H3z" /><path d="M12 10v4M12 17h.01" /></svg>
);
export const EditIcon = (p: P) => (
  <svg viewBox="0 0 24 24" {...base} {...p}><path d="M4 20h4L19 9l-4-4L4 16z" /><path d="M14 6l4 4" /></svg>
);
export const TrashIcon = (p: P) => (
  <svg viewBox="0 0 24 24" {...base} {...p}><path d="M4 7h16M10 11v6M14 11v6M6 7l1 13h10l1-13M9 7V4h6v3" /></svg>
);
export const PlusIcon = (p: P) => (
  <svg viewBox="0 0 24 24" {...base} {...p}><path d="M12 5v14M5 12h14" /></svg>
);
export const LockIcon = (p: P) => (
  <svg viewBox="0 0 24 24" {...base} {...p}><rect x="5" y="11" width="14" height="9" rx="2" /><path d="M8 11V8a4 4 0 0 1 8 0v3" /></svg>
);
export const EyeIcon = (p: P) => (
  <svg viewBox="0 0 24 24" {...base} {...p}><path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7S2 12 2 12z" /><circle cx="12" cy="12" r="3" /></svg>
);
export const FileIcon = (p: P) => (
  <svg viewBox="0 0 24 24" {...base} {...p}><path d="M6 3h8l4 4v14H6z" /><path d="M14 3v4h4" /></svg>
);
export const UploadIcon = (p: P) => (
  <svg viewBox="0 0 24 24" {...base} {...p}><path d="M12 16V4M7 9l5-5 5 5M4 20h16" /></svg>
);
export const CodeIcon = (p: P) => (
  <svg viewBox="0 0 24 24" {...base} {...p}><path d="M5 5h14v14H5z" /><path d="M9 9h6M9 13h6M9 17h3" /></svg>
);
