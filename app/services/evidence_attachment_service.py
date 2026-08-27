"""Real file storage for merchant evidence attachments (Phase 2).

Local disk only, local synthetic-data demonstration only -- not a claim of
production-grade document storage or malware scanning (see SECURITY.md).

Every file is validated before it touches disk:
- extension allowlist
- hard size cap
- a magic-byte sniff of the file's actual content, so a renamed executable
  can't pass as a PDF/image just because of its filename extension

The client-supplied filename is used only for display (`original_filename`)
-- it never appears in the storage path. The stored file always gets a
server-generated name, so there is no path-traversal surface here at all.
"""

from __future__ import annotations

import uuid
from pathlib import Path

from sqlalchemy.orm import Session

from app.db.models import EvidenceAttachment, EvidenceSubmission

STORAGE_DIR = Path("data/evidence_attachments")

MAX_SIZE_BYTES = 5 * 1024 * 1024  # 5 MB, per approved design decision

ALLOWED_EXTENSIONS = {"pdf", "txt", "png", "jpg", "jpeg"}

# First bytes of each allowed type, used to verify the upload's real content
# matches its claimed extension. Plain text has no reliable magic number, so
# .txt files are checked by trying to decode a small sample as UTF-8 instead.
_MAGIC_BYTES = {
    "pdf": [b"%PDF-"],
    "png": [b"\x89PNG\r\n\x1a\n"],
    "jpg": [b"\xff\xd8\xff"],
    "jpeg": [b"\xff\xd8\xff"],
}

# The stored content_type is always derived from the validated extension,
# never trusted from the client's claimed Content-Type header -- that
# header is attacker-controlled and unrelated to what was actually verified
# via the magic-byte check above.
_CONTENT_TYPE_BY_EXTENSION = {
    "pdf": "application/pdf",
    "txt": "text/plain",
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
}


class InvalidAttachmentError(Exception):
    """Raised for any attachment-validation failure. The upload is never
    written to disk when this is raised."""


def _extension_of(filename: str) -> str:
    return filename.rsplit(".", 1)[-1].lower() if "." in filename else ""


def validate_attachment(filename: str, content: bytes, content_type: str) -> str:
    """Validates a candidate attachment. Returns the normalized extension on
    success; raises InvalidAttachmentError with a safe, specific message
    otherwise. Takes already-read bytes (the caller is responsible for
    enforcing a read-size cap at the framework layer too, so an oversized
    upload is never fully buffered in memory unbounded)."""

    if not filename or not filename.strip():
        raise InvalidAttachmentError("A filename is required.")

    extension = _extension_of(filename)
    if extension not in ALLOWED_EXTENSIONS:
        raise InvalidAttachmentError(
            f"File type '.{extension}' is not allowed. Allowed types: {', '.join(sorted(ALLOWED_EXTENSIONS))}."
        )

    if len(content) == 0:
        raise InvalidAttachmentError("Uploaded file is empty.")
    if len(content) > MAX_SIZE_BYTES:
        raise InvalidAttachmentError(f"File exceeds the maximum allowed size of {MAX_SIZE_BYTES // (1024 * 1024)} MB.")

    if extension == "txt":
        try:
            content.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise InvalidAttachmentError("File claims to be .txt but is not valid UTF-8 text.") from exc
    else:
        signatures = _MAGIC_BYTES.get(extension, [])
        if not any(content.startswith(sig) for sig in signatures):
            raise InvalidAttachmentError(
                f"File content does not match its claimed '.{extension}' type -- upload rejected."
            )

    return extension


def save_attachment(
    session: Session,
    evidence: EvidenceSubmission,
    filename: str,
    content: bytes,
    claimed_content_type: str,
) -> EvidenceAttachment:
    """Validates and persists an attachment: writes the file to
    STORAGE_DIR under a generated name, then records an EvidenceAttachment
    row. Raises InvalidAttachmentError before anything is written if
    validation fails. claimed_content_type (the client's Content-Type
    header) is accepted only for symmetry with the upload request -- it is
    never trusted or stored; the persisted content_type is always derived
    from the validated, magic-byte-verified extension."""

    extension = validate_attachment(filename, content, claimed_content_type)

    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    stored_filename = f"attachment_{uuid.uuid4().hex}.{extension}"
    stored_path = STORAGE_DIR / stored_filename
    stored_path.write_bytes(content)

    attachment = EvidenceAttachment(
        attachment_id=f"attachment_{uuid.uuid4().hex[:16]}",
        evidence_id=evidence.evidence_id,
        original_filename=filename[:255],
        stored_filename=stored_filename,
        content_type=_CONTENT_TYPE_BY_EXTENSION[extension],
        size_bytes=len(content),
    )
    session.add(attachment)
    session.flush()
    return attachment


def read_attachment_bytes(attachment: EvidenceAttachment) -> bytes:
    path = STORAGE_DIR / attachment.stored_filename
    return path.read_bytes()
