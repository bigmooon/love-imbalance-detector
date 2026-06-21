"""분석 모듈이 Streamlit 없이 import 가능한지 검증."""
import subprocess
import sys


def test_models_importable_without_streamlit():
    """models.hugging_face가 streamlit을 import하지 않아야 한다."""
    code = (
        "import sys; import models.hugging_face; "
        "assert 'streamlit' not in sys.modules, 'streamlit imported!'"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_parser_importable_without_streamlit():
    code = (
        "import sys; import utils.kakao_parser; "
        "assert 'streamlit' not in sys.modules, 'streamlit imported!'"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_weight_presets_moved_to_features():
    from features.presets import WEIGHT_PRESETS
    assert "기본" in WEIGHT_PRESETS
    assert WEIGHT_PRESETS["기본"] == {"dominance": None, "dependence": None}
    assert "답장속도 중시" in WEIGHT_PRESETS
    assert "감정 중시" in WEIGHT_PRESETS
