# Phase 2 Design — Evidence Attachments (as-built)

**Status: implemented, on branch `phase-2-auth-design`.** Written after the fact (unlike `docs/PHASE_2_AUTH_DESIGN.md`, this feature was small enough and closely bounded enough by the prior auth work that implementation and design documentation happened together, with the two real open questions — max size and malware scanning — confirmed with the user before writing code).

## 1. Goal

Replace the current filename-string-only evidence references with a real file upload, now that Phase 2 auth exists to make uploads attributable to a specific merchant. Per `CLAUDE.md`'s Phase 2 roadmap: "Add evidence attachments safely."

## 2. Decisions (confirmed with the user before implementation)

1. **Max size: 5 MB** per attachment.
2. **No malware/AV scanning** — out of scope for this local demo, documented as a known limitation (extension allowlist + magic-byte content check + generated filenames is the mitigation instead).
3. **A real `st.file_uploader` is added to the dashboard**, which required updating `tests/test_dashboard_safety.py::test_merchant_response_page_has_no_file_uploader` (renamed to `test_merchant_response_page_file_uploader_is_type_restricted`) since it previously asserted zero file-upload capability existed anywhere in the dashboard — an assertion this feature deliberately supersedes, confirmed with the user first.

## 3. Data boundary and safety

- **Storage**: local filesystem only, `data/evidence_attachments/` (gitignored, never committed).
- **No path traversal surface**: the client-supplied filename is used only for display (`original_filename`, capped at 255 chars). The file on disk is always saved under a server-generated name (`attachment_<uuid>.<ext>`) — the client's filename never touches a filesystem path.
- **Type validation, two layers**:
  1. Extension allowlist: `pdf`, `txt`, `png`, `jpg`, `jpeg`.
  2. Magic-byte content check: the file's actual first bytes must match its claimed extension (e.g. `%PDF-` for `.pdf`, PNG/JPEG signatures for images, valid UTF-8 for `.txt`). This is the check that stops a renamed executable from passing as a PDF just because of its filename — verified in tests and in a live end-to-end smoke test (a `MZ`-header fake `.pdf` was rejected with `"File content does not match its claimed '.pdf' type"`).
- **`Content-Type` is never trusted from the client.** The stored (and later served) `content_type` is always derived from the validated extension, not the client's `Content-Type` header.
- **Size cap enforced twice**: once at the route layer (`await file.read(MAX_UPLOAD_READ_BYTES)` reads at most one byte over the 5 MB cap, so an oversized upload is never fully buffered), and once in the service layer's own check.
- **Access control**: upload requires `role="merchant"` and `case.merchant_id == user.merchant_id` (404, not 403, on mismatch — same pattern as evidence submission). Download requires any authenticated role; a `merchant`-role caller is scoped to their own `merchant_id` exactly like every other case read.
- **Header-injection hardening**: the download route strips `\r`, `\n`, and `"` from the client-supplied `original_filename` before it goes into the `Content-Disposition` response header, since that value is otherwise attacker-controlled.

## 4. Data model

New table `evidence_attachments` (`app/db/models.py`): `attachment_id`, `evidence_id` (fk → `evidence_submissions`), `original_filename`, `stored_filename` (unique), `content_type`, `size_bytes`, `uploaded_at`.

## 5. API surface

- `POST /cases/{case_id}/evidence/{evidence_id}/attachments` — multipart upload, `role="merchant"` only, ownership-checked. Returns `{"attachment": {...}, "new_audit_event": {...}, "synthetic_data_notice": "..."}`.
- `GET /cases/{case_id}/evidence/{evidence_id}/attachments/{attachment_id}` — download, any authenticated role (merchant-scoped for merchant callers). Returns the raw file bytes with `Content-Disposition: attachment`.
- `EvidenceSubmissionSummary` (in `GET /cases/{case_id}`) gained an `attachments: list[EvidenceAttachmentSummary]` field, so attachments are visible in case detail without a separate call.
- A new audit event type, `EVIDENCE_ATTACHMENT_UPLOADED`, is recorded on every successful upload with safe metadata only (attachment id, filename, content type, size — never raw file bytes).

## 6. Dashboard integration

- `dashboard/components/evidence_form.py` — the Merchant Response page gained `st.file_uploader(type=["pdf","txt","png","jpg","jpeg"])`. On submit, the evidence text is always submitted first; the file (if any) is uploaded as a second call. If the text submission succeeds but the attachment is rejected, the page says so plainly ("Evidence text was submitted, but the file attachment was rejected: ...") rather than implying total failure.
- `dashboard/components/case_detail.py` — the "Evidence checklist" tab now also shows a "Submitted evidence" section listing each evidence submission's references and attachments, with a Download button per attachment (fetches bytes through the authenticated API client, then offers `st.download_button` to save).
- `dashboard/api_client.py` gained `upload_attachment()` (multipart) and `download_attachment()` (raw bytes) — both bypass the shared JSON-only `_request()` helper since neither request/response is JSON.

## 7. Tests

- `tests/test_evidence_attachment_service.py` (12 tests) — extension allowlist, size cap, magic-byte mismatch (including the renamed-executable case), non-UTF-8 `.txt` rejection, case-insensitive extension matching.
- `tests/test_api_evidence_attachments.py` (9 tests) — upload success, disallowed type, content/extension mismatch, role enforcement (merchant-only), cross-merchant upload/download rejection (404), unknown attachment id (404), and that an uploaded attachment appears in `GET /cases/{case_id}`.
- `tests/test_dashboard_safety.py` — updated (not removed) to assert the uploader is type-restricted and no camera/webcam capture exists, rather than asserting zero upload capability.
- Full suite passing after this change (see `docs/MILESTONE_9_AUTH.md`'s sibling report, or the top-level test count in `README.md`, for the current number).

## 8. Manual end-to-end verification performed

Beyond the automated tests, a live smoke test was run against the real running FastAPI backend and a real seeded case: text evidence submitted, a real PDF uploaded and confirmed byte-for-byte identical on download, a `MZ`-header file renamed to `.pdf` correctly rejected, an unauthenticated download request correctly rejected, and the attachment's metadata confirmed visible in `GET /cases/{case_id}`. The temporary demo account created for this test was removed afterward, and the local demo database was reseeded to a clean state.

## 9. Known limitations (not fabricated, flagged instead)

- No malware/antivirus scanning (confirmed out of scope for this local demo).
- No thumbnail/preview generation for images — files are download-only, not previewed inline.
- No storage quota per merchant/case — a merchant could in principle upload many 5 MB files across many evidence submissions; no aggregate cap exists yet.
- Local filesystem storage only; no cloud/object storage, no encryption at rest.
