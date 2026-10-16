import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)
LAST_YEAR = 2025      # 수업 기준 기간의 마지막 해
MIN_DAYS = 300        # 관측일이 이보다 적은 해는 제외

st.set_page_config(page_title="기온 예측기", page_icon="🌡️", layout="centered")


@st.cache_data(show_spinner="데이터를 불러오는 중...")
def load_yearly() -> pd.DataFrame:
    df = pd.read_csv(DATA_URL, encoding="utf-8")
    df.columns = [str(c).strip() for c in df.columns]
    df["날짜"] = pd.to_datetime(df["날짜"].astype(str).str.strip(), errors="coerce")
    df["평균기온"] = pd.to_numeric(df["평균기온"], errors="coerce")
    df = df.dropna(subset=["날짜", "평균기온"])
    df = df[df["날짜"].dt.year <= LAST_YEAR]
    df["연도"] = df["날짜"].dt.year

    yearly = (
        df.groupby("연도")
        .agg(연평균기온=("평균기온", "mean"), 관측일수=("날짜", "nunique"))
        .reset_index()
    )
    return yearly[yearly["관측일수"] >= MIN_DAYS].reset_index(drop=True)


st.title("🌡️ 기온 예측기")
st.caption("서울의 연평균기온에 직선(선형 회귀)을 맞춰 원하는 해의 기온을 예상해 봅니다.")

try:
    yearly = load_yearly()
except Exception as e:
    st.error(f"데이터를 불러오지 못했어요: {e}")
    st.stop()

if len(yearly) < 2:
    st.error("회귀 직선을 만들 데이터가 부족합니다.")
    st.stop()

x = yearly["연도"].to_numpy()
y = yearly["연평균기온"].to_numpy()
slope, intercept = np.polyfit(x, y, 1)
r = np.corrcoef(x, y)[0, 1]

# 직선을 만든 데이터 정보
c1, c2, c3 = st.columns(3)
c1.metric("사용한 해의 개수", f"{len(yearly)}개")
c2.metric("시작 연도", f"{x.min()}년")
c3.metric("끝 연도", f"{x.max()}년")
st.write(
    f"{LAST_YEAR}년 이후 데이터와 관측일이 {MIN_DAYS}일 미만인 해는 제외했습니다."
)

st.metric("상관계수 (r)", f"{r:.3f}")

# 최근 20년 기울기 (끝 연도 기준 20년 구간)
RECENT_YEARS = 20
recent = yearly[yearly["연도"] > x.max() - RECENT_YEARS]
if len(recent) >= 2:
    slope_recent, intercept_recent = np.polyfit(
        recent["연도"].to_numpy(), recent["연평균기온"].to_numpy(), 1
    )
else:
    slope_recent = intercept_recent = None


def big_card(title: str, sub: str, value: float | None) -> str:
    text = "계산 불가" if value is None else f"{value * 100:+.2f} ℃"
    return f"""
    <div style="text-align:center; padding:1rem; border-radius:12px;
                border:1px solid rgba(128,128,128,0.35); height:100%;">
      <div style="font-size:1.05rem; font-weight:600;">{title}</div>
      <div style="font-size:0.85rem; opacity:0.7; margin-bottom:0.3rem;">{sub}</div>
      <div style="font-size:2.6rem; font-weight:700; line-height:1.2;">{text}</div>
      <div style="font-size:0.85rem; opacity:0.7;">100년에 오르는 기온</div>
    </div>
    """


st.subheader("100년에 몇 도 오를까?")
a, b = st.columns(2)
a.markdown(
    big_card("전체 기간", f"{x.min()}~{x.max()}년 · {len(yearly)}개 해", slope),
    unsafe_allow_html=True,
)
b.markdown(
    big_card(
        f"최근 {RECENT_YEARS}년",
        f"{recent['연도'].min()}~{recent['연도'].max()}년 · {len(recent)}개 해",
        slope_recent,
    ),
    unsafe_allow_html=True,
)
if slope_recent is not None and slope > 0 and slope_recent > 0:
    st.write(
        f"최근 {RECENT_YEARS}년의 상승 속도는 전체 기간의 약 "
        f"**{slope_recent / slope:.1f}배**입니다. "
        "(기울기 = 연간 변화량 × 100)"
    )
    st.caption("20년은 자료가 적고 해마다 들쭉날쭉해서, 기울기가 크게 흔들릴 수 있어요.")

# 연도 슬라이더와 예측값
year = st.slider("예상할 연도", min_value=1900, max_value=2100, value=2050, step=1)
pred = slope * year + intercept

st.markdown(
    f"""
    <div style="text-align:center; padding:1.2rem; margin:0.5rem 0 1rem;
                border-radius:12px; border:1px solid rgba(128,128,128,0.35);">
      <div style="font-size:1.1rem; opacity:0.8;">{year}년의 예상 연평균기온</div>
      <div style="font-size:3.6rem; font-weight:700; line-height:1.2;">{pred:.2f} ℃</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# 그래프
x_line = np.array([min(x.min(), year), max(x.max(), year)])
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=x, y=y, mode="markers", name="연평균기온",
        marker=dict(size=8, opacity=0.8),
        hovertemplate="%{x}년<br>%{y:.2f} ℃<extra></extra>",
    )
)
fig.add_trace(
    go.Scatter(
        x=x_line, y=slope * x_line + intercept, mode="lines",
        name=f"회귀 직선 (r = {r:.3f})", line=dict(width=3, color="crimson"),
    )
)
if slope_recent is not None:
    xr = recent["연도"].to_numpy()
    fig.add_trace(
        go.Scatter(
            x=[xr.min(), xr.max()],
            y=[slope_recent * xr.min() + intercept_recent,
               slope_recent * xr.max() + intercept_recent],
            mode="lines", name=f"최근 {RECENT_YEARS}년 직선",
            line=dict(width=3, dash="dash", color="royalblue"),
        )
    )
fig.add_trace(
    go.Scatter(
        x=[year], y=[pred], mode="markers", name=f"{year}년 예측",
        marker=dict(size=16, symbol="star", color="gold", line=dict(width=1.5, color="black")),
        hovertemplate="%{x}년<br>예측 %{y:.2f} ℃<extra></extra>",
    )
)
fig.update_layout(
    title="서울 연평균기온 산점도와 회귀 직선",
    xaxis_title="연도", yaxis_title="연평균기온 (℃)",
    legend=dict(orientation="h", yanchor="bottom", y=-0.35),
    margin=dict(l=10, r=10, t=50, b=10),
)
st.plotly_chart(fig, use_container_width=True)

st.info(
    f"회귀식: 기온 = {slope:.4f} × 연도 {intercept:+.2f}  \n"
    f"관측 기간({x.min()}~{x.max()}년) 밖의 연도는 직선을 그대로 늘려 계산한 값이라 "
    "실제와 차이가 클 수 있어요."
)
