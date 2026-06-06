import pytest


@pytest.fixture
def sample_payload_dict():
    box = {"lo": 0.2, "q1": 0.5, "median": 1.0, "q3": 2.0, "hi": 10.0}
    return {
        "me": "지언", "partner": "민수",
        "dominance_index": 0.62, "dependence_index": 0.71, "balance": 0.91,
        "radar": {
            "categories": ["선톡 비율", "대화 종료", "메시지 비율", "글자 비율", "답장 속도", "더블텍스트", "QA 성의도"],
            "me": [0.6, 0.5, 0.55, 0.6, 0.7, 0.4, 0.5],
            "partner": [0.4, 0.5, 0.45, 0.4, 0.3, 0.6, 0.5],
        },
        "participation": {
            "message_count_ratio": 0.55, "char_count_ratio": 0.6,
            "avg_length_me": 18.2, "avg_length_partner": 12.1,
        },
        "timeline": [{"week": "2025-01-06", "me": 12, "partner": 18}],
        "emotion": {
            "me": {"joy": 0.4, "anger": 0.1, "sadness": 0.1, "anxiety": 0.2, "hurt": 0.1, "embarrass": 0.1},
            "partner": {"joy": 0.5, "anger": 0.1, "sadness": 0.1, "anxiety": 0.1, "hurt": 0.1, "embarrass": 0.1},
            "joy_gap": -0.1, "negative_gap": 0.1,
        },
        "reply_time": {
            "me_median_sec": 35.0, "partner_median_sec": 180.0,
            "me_box": box, "partner_box": box,
        },
        "double_text": {"me": 0.21, "partner": 0.13},
        "initiation_ratio": 0.7,
        "qa_sincerity": {
            "avg_sincerity": 0.55, "my_sincerity": 0.6, "partner_sincerity": 0.5,
            "pairs": [{"questioner": "민수", "question": "뭐해?", "answerer": "지언", "answer": "일해", "score": 0.3}],
        },
        "llm": None,
        "llm_error": None,
    }
