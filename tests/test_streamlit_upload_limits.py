"""Test the upload guard without launching Streamlit or loading models."""

import importlib.util
from pathlib import Path
from unittest.mock import MagicMock, Mock

import pytest


@pytest.fixture
def app(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "scripts" / "streamlit_app.py"
    spec = importlib.util.spec_from_file_location("ragforge_streamlit_test_app", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    ui = MagicMock()
    ui.text_input.side_effect = ["qwen2.5:3b", ""]
    ui.button.return_value = False
    ui.session_state = {}
    monkeypatch.setattr(module, "st", ui)
    monkeypatch.setattr(module, "build_pdf_session", Mock())
    return module, ui


def test_oversized_upload_is_rejected_before_copying_bytes(app):
    module, ui = app
    uploaded = Mock()
    uploaded.size = module.MAX_PDF_BYTES + 1
    ui.file_uploader.return_value = uploaded
    ui.session_state.update(
        {
            "pdf_session": object(),
            "session_key": "previous-session",
            "last_result": {"answer": "previous-answer"},
            "unrelated_setting": "keep",
        }
    )

    module.main()

    uploaded.getvalue.assert_not_called()
    module.build_pdf_session.assert_not_called()
    ui.error.assert_called_once()
    assert "upload limit" in ui.error.call_args.args[0]
    assert ui.session_state == {"unrelated_setting": "keep"}
    assert ui.button.call_count == 1


@pytest.mark.parametrize("size_offset", [-1, 0])
def test_upload_at_or_below_limit_passes_guard(app, size_offset):
    module, ui = app
    uploaded = Mock()
    uploaded.size = module.MAX_PDF_BYTES + size_offset
    uploaded.name = "sample.pdf"
    uploaded.getvalue.return_value = b"sample PDF bytes"
    ui.file_uploader.return_value = uploaded

    module.main()

    uploaded.getvalue.assert_called_once_with()
    ui.error.assert_not_called()
    module.build_pdf_session.assert_not_called()
    assert ui.button.call_count == 2
