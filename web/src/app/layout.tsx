import type { Metadata } from "next";

import "./globals.css";
import { Sidebar } from "@/components/Sidebar";

export const metadata: Metadata = {
  title: "Notewell - AI medical scribe",
  description: "Visit audio to a grounded SOAP note with ICD-10-CM and E/M suggestions, PHI masked before any cloud model. Synthetic demo.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" className="h-full antialiased">
      <body className="flex min-h-full text-zinc-900">
        <Sidebar />
        <main className="min-w-0 flex-1">{children}</main>
      </body>
    </html>
  );
}
