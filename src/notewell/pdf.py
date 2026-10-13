"""PDF rendering of a note (reportlab): header, SOAP sections, medications, allergies, codes, after-visit summary,
the signature line, and the demo disclaimer on every page."""

from __future__ import annotations

import io
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import ListFlowable, ListItem, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from notewell.models import SECTIONS, Note, Visit

DISCLAIMER = "Synthetic demo data. Notewell is not a medical device and not for clinical use."


def render_pdf(visit: Visit, note: Note) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=LETTER, leftMargin=0.8 * inch, rightMargin=0.8 * inch, topMargin=0.7 * inch, bottomMargin=0.8 * inch,
        title=f"Progress note - {visit.title}", author=visit.clinician.name,
    )  # fmt: skip
    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=styles["Heading1"], fontSize=16, spaceAfter=4, textColor=colors.HexColor("#12355b"))
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=12, spaceBefore=10, spaceAfter=3, textColor=colors.HexColor("#12355b"))
    body = ParagraphStyle("body", parent=styles["BodyText"], fontSize=9.5, leading=12.5)
    small = ParagraphStyle("small", parent=body, fontSize=8, textColor=colors.HexColor("#555555"))
    story: list[object] = [Paragraph("Progress note", h1)]
    p = visit.patient
    meta = [
        ["Patient", f"{p.name}  (DOB {p.dob or 'n/a'}, MRN {p.mrn or 'n/a'})", "Visit", f"{visit.visit_date}, {visit.visit_type}"],
        ["Clinician", visit.clinician.name, "Specialty", visit.specialty.replace("_", " ").title()],
    ]
    table = Table(meta, colWidths=[0.8 * inch, 3.2 * inch, 0.8 * inch, 2.1 * inch])
    table.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), 8.5), ("TEXTCOLOR", (0, 0), (0, -1), colors.grey),
                               ("TEXTCOLOR", (2, 0), (2, -1), colors.grey), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]))  # fmt: skip
    story += [table, Spacer(1, 6)]
    for section in SECTIONS:
        story.append(Paragraph(section.capitalize(), h2))
        items = [ListItem(Paragraph(escape(s.text), body), leftIndent=10) for s in note.section(section)]
        if items:
            story.append(ListFlowable(items, bulletType="bullet", start="•", leftIndent=12))
        if section == "subjective":
            meds = "; ".join(" ".join(x for x in (m.name, m.dose, m.frequency, f"({m.status})") if x) for m in note.medications)
            allergies = "; ".join(f"{a.substance} ({a.reaction})" if a.reaction else a.substance for a in note.allergies)
            story.append(Paragraph(f"<b>Medications:</b> {escape(meds) or 'none documented'}", body))
            story.append(Paragraph(f"<b>Allergies:</b> {escape(allergies) or 'none reported'}", body))
    accepted = [c for c in note.icd10 if c.accepted]
    story.append(Paragraph("Codes", h2))
    rows = [[c.code, c.description] for c in accepted]
    if note.em and note.em.accepted is not False:
        rows.append([note.em.code, note.em.description + " (CPT(R) code; descriptor paraphrased)"])
    rows += [[c.code, c.description] for c in note.procedures if c.accepted]
    if rows:
        codes = Table([["Code", "Description"], *rows], colWidths=[0.9 * inch, 6.0 * inch])
        codes.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), 8.5), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef6")),
                                   ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#c8d2de"))]))  # fmt: skip
        story.append(codes)
    else:
        story.append(Paragraph("No codes accepted.", body))
    if note.after_visit_summary:
        story += [Paragraph("After-visit summary (for the patient)", h2), Paragraph(escape(note.after_visit_summary), body)]
    signed = (
        f"Electronically signed by {escape(note.signed_by or '')} on {note.signed_at:%Y-%m-%d %H:%M} UTC."
        if note.signed_at
        else "DRAFT - not signed."
    )
    story += [Spacer(1, 10), Paragraph(f"<b>{signed}</b>", body),
              Paragraph(f"Drafted with AI assistance ({escape(note.model)}), reviewed and edited by the clinician.", small)]  # fmt: skip

    def footer(canvas: object, _doc: object) -> None:
        c = canvas
        c.saveState()  # type: ignore[attr-defined]
        c.setFont("Helvetica", 7.5)  # type: ignore[attr-defined]
        c.setFillColor(colors.HexColor("#9a3412"))  # type: ignore[attr-defined]
        c.drawString(0.8 * inch, 0.45 * inch, DISCLAIMER)  # type: ignore[attr-defined]
        c.restoreState()  # type: ignore[attr-defined]

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return buffer.getvalue()
