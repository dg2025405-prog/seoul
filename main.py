import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# --------------------------------------------------
# 기본 설정
# --------------------------------------------------
st.set_page_config(
    page_title="기온 예측기",
    page_icon="🌡️",
    layout="wide"
)

DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)

# --------------------------------------------------
# 제목
# --------------------------------------------------
st.title("🌡️ 기온 예측기")
st.caption("서울의 과거 연평균기온을 바탕으로 선택한 연도의 예상 평균기온을 계산합니다.")

# --------------------------------------------------
# 데이터 불러오기
# --------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")

    # 날짜 변환
    df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")

    # 평균기온 숫자 변환
    df["평균기온"] = pd.to_numeric(df["평균기온"], errors="coerce")

    # 필요한 데이터만 사용
    df = df.dropna(subset=["날짜", "평균기온"])

    # 연도 추출
    df["연도"] = df["날짜"].dt.year

    return df


try:
    df = load_data()
except Exception as e:
    st.error("데이터를 불러오는 중 오류가 발생했습니다.")
    st.stop()

# --------------------------------------------------
# 연도별 통계 계산
# --------------------------------------------------
yearly = (
    df.groupby("연도")
    .agg(
        평균기온=("평균기온", "mean"),
        관측일수=("평균기온", "count")
    )
    .reset_index()
)

# --------------------------------------------------
# 기준 기간 적용
# 1. 2025년까지
# 2. 관측일수 300일 이상
# --------------------------------------------------
yearly = yearly[
    (yearly["연도"] <= 2025) &
    (yearly["관측일수"] >= 300)
].copy()

yearly = yearly.sort_values("연도").reset_index(drop=True)

# 데이터가 충분하지 않은 경우
if len(yearly) < 2:
    st.error("회귀분석을 수행하기에 충분한 연도별 데이터가 없습니다.")
    st.stop()

# --------------------------------------------------
# 선형 회귀
# --------------------------------------------------
x = yearly["연도"].to_numpy(dtype=float)
y = yearly["평균기온"].to_numpy(dtype=float)

slope, intercept = np.polyfit(x, y, 1)

# 회귀식에 따른 값
yearly["회귀기온"] = slope * yearly["연도"] + intercept

# 상관계수
correlation = np.corrcoef(x, y)[0, 1]

# 회귀선 표시용 범위
line_x = np.linspace(x.min(), x.max(), 200)
line_y = slope * line_x + intercept

# --------------------------------------------------
# 연도 선택
# --------------------------------------------------
st.subheader("📅 예상 연도 선택")

selected_year = st.slider(
    "연도",
    min_value=1900,
    max_value=2100,
    value=2025,
    step=1
)

predicted_temperature = slope * selected_year + intercept

# --------------------------------------------------
# 선택한 연도의 예상 기온
# --------------------------------------------------
st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "선택한 연도",
        f"{selected_year}년"
    )

with col2:
    st.metric(
        "예상 연평균기온",
        f"{predicted_temperature:.2f} °C"
    )

with col3:
    st.metric(
        "상관계수",
        f"{correlation:.3f}"
    )

# --------------------------------------------------
# 회귀 정보
# --------------------------------------------------
st.markdown("---")
st.subheader("📊 회귀분석 정보")

info1, info2, info3 = st.columns(3)

with info1:
    st.markdown("### 사용한 연도 수")
    st.markdown(f"## {len(yearly)}개")

with info2:
    st.markdown("### 시작 연도")
    st.markdown(f"## {int(yearly['연도'].min())}년")

with info3:
    st.markdown("### 끝 연도")
    st.markdown(f"## {int(yearly['연도'].max())}년")

st.info(
    f"분석 기준: **2025년까지**, 연간 **관측일수 300일 이상**인 연도만 사용했습니다."
)

# --------------------------------------------------
# 회귀식
# --------------------------------------------------
st.write(
    f"**회귀식:** 연평균기온 = "
    f"{slope:.5f} × 연도 + {intercept:.2f}"
)

# --------------------------------------------------
# Plotly 산점도 + 회귀 직선
# --------------------------------------------------
fig = go.Figure()

# 실제 연평균기온 산점도
fig.add_trace(
    go.Scatter(
        x=yearly["연도"],
        y=yearly["평균기온"],
        mode="markers",
        name="실제 연평균기온",
        text=[
            f"{year}년<br>"
            f"평균기온: {temp:.2f}°C<br>"
            f"관측일수: {days}일"
            for year, temp, days in zip(
                yearly["연도"],
                yearly["평균기온"],
                yearly["관측일수"]
            )
        ],
        hovertemplate="%{text}<extra></extra>",
        marker=dict(
            size=7,
            opacity=0.75
        )
    )
)

# 회귀 직선
fig.add_trace(
    go.Scatter(
        x=line_x,
        y=line_y,
        mode="lines",
        name="회귀 직선",
        line=dict(
            width=3
        )
    )
)

# 선택한 연도의 예측값
fig.add_trace(
    go.Scatter(
        x=[selected_year],
        y=[predicted_temperature],
        mode="markers",
        name="선택 연도 예상값",
        marker=dict(
            size=15,
            symbol="star"
        ),
        hovertemplate=(
            f"{selected_year}년 예상 연평균기온"
            f"<br>{predicted_temperature:.2f}°C"
            "<extra></extra>"
        )
    )
)

fig.update_layout(
    title="서울 연도별 연평균기온과 회귀 직선",
    xaxis_title="연도",
    yaxis_title="연평균기온 (°C)",
    hovermode="closest",
    height=600,
    template="plotly_white",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0
    )
)

fig.update_xaxes(
    dtick=10
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# --------------------------------------------------
# 데이터 표
# --------------------------------------------------
with st.expander("📋 회귀에 사용된 연도별 데이터 보기"):
    display_df = yearly.copy()

    display_df["평균기온"] = display_df["평균기온"].round(2)
    display_df["회귀기온"] = display_df["회귀기온"].round(2)

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )

# --------------------------------------------------
# 안내
# --------------------------------------------------
st.caption(
    "※ 1900~2100년 슬라이더의 값은 과거 자료로 계산한 선형 회귀식을 "
    "해당 연도까지 연장하여 계산한 예상값입니다."
)
