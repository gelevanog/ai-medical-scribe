# Notewell: an AI medical scribe that turns a visit recording into a grounded SOAP note with codes

**Record or upload a clinician–patient conversation. Notewell transcribes it on the clinic's own CPU, masks every identifier before any cloud model sees the text, drafts a SOAP note, a patient-friendly after-visit summary and ICD-10-CM / E/M coding suggestions, ties every statement to a quote from the conversation, flags what it cannot back up, and waits for the clinician to review and sign. Signed notes export as FHIR R4 and PDF.**

[![CI](https://github.com/gelevanog/ai-medical-scribe/actions/workflows/ci.yml/badge.svg)](https://github.com/gelevanog/ai-medical-scribe/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.143-009688?logo=fastapi&logoColor=white)
![Next.js](https://img.shields.io/badge/Next.js-16-000000?logo=nextdotjs&logoColor=white)
![STT](https://img.shields.io/badge/STT-faster--whisper%20on%20CPU-FFD21E)
![FHIR](https://img.shields.io/badge/FHIR-R4-E44D26)
![MCP](https://img.shields.io/badge/MCP-server-6E56CF)
![mypy strict](https://img.shields.io/badge/mypy-strict-2a6db2)
![License: MIT](https://img.shields.io/badge/License-MIT-green)

<!-- demo-video -->

![The review screen: transcript with speaker roles, the SOAP note with the evidence of the hovered statement highlighted in the transcript, ICD-10-CM and E/M suggestions with the rule-based MDM estimate, and the sign-off checklist](docs/screenshots/hero.png)

<sub>A synthetic heart-failure visit drafted by the free model `nvidia/nemotron-3-super-120b-a12b:free`. Hovering "Denies chest pain" highlights the question and the answer it came from (turns 12 and 13); the medication list shows two checks a clinician must look at (a dose corrected during the conversation, a dose nobody said); the sign-off stays disabled until every section is accepted and every issue resolved.</sub>

> **Not a medical device. Demo only, not for clinical use.** Every patient, clinician, name, number and address in this repository is synthetic. Notewell is a portfolio project that shows how such a system is built and measured; it is not cleared or approved for clinical use and it makes no claim of HIPAA compliance (see [HIPAA-minded design](#hipaa-minded-design)).

RESULTS_PLACEHOLDER
