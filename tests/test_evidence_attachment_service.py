"""Unit tests for app/services/evidence_attachment_service.py -- the
validation and storage logic for real evidence-attachment uploads
(Phase 2)."""

from __future__ import annotations

import pytest

from app.services.evidence_attachment_service import InvalidAttachmentError, validate_attachment

PDF_BYTES = b"%PDF-1.4\n%rest of a fake pdf body"
PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"rest of a fake png body"
JPEG_BYTES = b"\xff\xd8\xff" + b"rest of a fake jpeg body"
TXT_BYTES = "a plain text explanation".encode("utf-8")


def test_valid_pdf_is_accepted():
    assert validate_attachment("invoice.pdf", PDF_BYTES, "application/pdf") == "pdf"


def test_valid_png_is_accepted():
    assert validate_attachment("photo.png", PNG_BYTES, "image/png") == "png"


def test_valid_jpeg_is_accepted():
    assert validate_attachment("photo.jpeg", JPEG_BYTES, "image/jpeg") == "jpeg"


def test_valid_txt_is_accepted():
    assert validate_attachment("note.txt", TXT_BYTES, "text/plain") == "txt"


def test_disallowed_extension_is_rejected():
    with pytest.raises(InvalidAttachmentError, match="not allowed"):
        validate_attachment("script.exe", b"MZ\x90\x00", "application/octet-stream")


def test_no_extension_is_rejected():
    with pytest.raises(InvalidAttachmentError, match="not allowed"):
        validate_attachment("noextension", b"whatever", "application/octet-stream")


def test_empty_file_is_rejected():
    with pytest.raises(InvalidAttachmentError, match="empty"):
        validate_attachment("empty.pdf", b"", "application/pdf")


def test_oversized_file_is_rejected():
    oversized = PDF_BYTES + b"0" * (5 * 1024 * 1024)
    with pytest.raises(InvalidAttachmentError, match="exceeds the maximum"):
        validate_attachment("big.pdf", oversized, "application/pdf")


def test_content_not_matching_claimed_extension_is_rejected():
    """The classic attack this check exists for: an executable renamed with
    a safe-looking extension."""
    fake_pdf = b"MZ\x90\x00\x03\x00\x00\x00"  # actual Windows PE header
    with pytest.raises(InvalidAttachmentError, match="does not match"):
        validate_attachment("totally_a_pdf.pdf", fake_pdf, "application/pdf")


def test_non_utf8_txt_is_rejected():
    with pytest.raises(InvalidAttachmentError, match="not valid UTF-8"):
        validate_attachment("note.txt", b"\xff\xfe\x00\x01", "text/plain")


def test_blank_filename_is_rejected():
    with pytest.raises(InvalidAttachmentError, match="filename is required"):
        validate_attachment("", PDF_BYTES, "application/pdf")


def test_extension_check_is_case_insensitive():
    assert validate_attachment("INVOICE.PDF", PDF_BYTES, "application/pdf") == "pdf"
