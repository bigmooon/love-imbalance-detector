# llm/ui.py
import streamlit as st


def _score_row(label: str, comp: dict):
  icon = "✅ AI도 동의" if comp["agree"] else "⚠️ 관점 차이"
  c1, c2, c3 = st.columns(3)
  c1.metric(f"{label} · 규칙(BERT)", f"{comp['tier1']:.2f}")
  c2.metric(f"{label} · LLM", f"{comp['llm']:.2f}", delta=f"{comp['delta']:+.2f}")
  c3.metric("일치 여부", icon)


def _evidence_cards(title: str, windows: list[dict]):
  st.markdown(f"**{title}**")
  for w in windows[:4]:
    st.markdown(
      f"<div style='background:#f7f7fb;border-radius:10px;padding:10px 14px;margin-bottom:8px;'>"
      f"<span style='color:#888;font-size:12px;'>유사도 {w['sim']:.2f}</span><br>"
      f"<span style='font-size:14px;white-space:pre-line;'>{w['text']}</span></div>",
      unsafe_allow_html=True,
    )


def render_llm_section(llm_result, me: str, partner: str):
  """analysis_result['llm']을 받아 AI 심층 분석 섹션을 그린다.

  llm_result 형태:
    None                                   → 키 없음/스킵
    {"error": str}                         → 호출 실패
    {"judgment","comparison","retrieved"}  → 정상
  """
  st.divider()
  st.subheader("🤖 AI 심층 분석 (LLM)")

  if llm_result is None:
    st.info("💡 사이드바에 OpenAI API 키를 입력하면 LLM 심층 분석이 활성화됩니다. (규칙 기반 결과는 위에 그대로 유지됩니다)")
    return

  if "error" in llm_result:
    st.warning(f"⚠️ LLM 분석을 완료하지 못했어요: {llm_result['error']} (규칙 기반 결과는 정상입니다)")
    return

  comparison = llm_result["comparison"]
  judgment = llm_result["judgment"]
  retrieved = llm_result["retrieved"]

  st.caption(f"LLM 신뢰도: {judgment.confidence:.0%}")
  _score_row("지배성", comparison["dominance"])
  _score_row("의존도", comparison["dependence"])

  with st.expander("🔍 LLM이 주목한 근거 대화"):
    _evidence_cards("지배성 근거", retrieved["dominance"])
    _evidence_cards("의존도 근거", retrieved["dependence"])

  st.markdown("### 📝 AI 진단 리포트")
  st.markdown(judgment.report)
