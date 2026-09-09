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
# 데이터 불러오기
# --------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")

    df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")
    df["평균기온"] = pd.to_numeric(df["평균기온"], errors="coerce")

    df = df.dropna(subset=["날짜", "평균기온"])
    df["연도"] = df["날짜"].dt.year

    return df


# --------------------------------------------------
# 데이터 처리
# --------------------------------------------------
try:
    df = load_data()
except Exception:
    st.error("서울 기온 데이터를 불러오지 못했습니다.")
    st.stop()


# 연도별 평균기온 + 관측일수
yearly = (
    df.groupby("연도")
    .agg(
        평균기온=("평균기온", "mean"),
        관측일수=("평균기온", "count")
    )
    .reset_index()
)

# 2025년까지 + 관측일수 300일 이상
yearly = yearly[
    (yearly["연도"] <= 2025) &
    (yearly["관측일수"] >= 300)
].copy()

yearly = yearly.sort_values("연도").reset_index(drop=True)

if len(yearly) < 2:
    st.error("회귀분석에 필요한 데이터가 충분하지 않습니다.")
    st.stop()


# --------------------------------------------------
# 전체 기간 회귀분석
# --------------------------------------------------
x_all = yearly["연도"].to_numpy(dtype=float)
y_all = yearly["평균기온"].to_numpy(dtype=float)

slope_all, intercept_all = np.polyfit(x_all, y_all, 1)

# 100년당 변화량
slope_all_100 = slope_all * 100

correlation = np.corrcoef(x_all, y_all)[0, 1]


# --------------------------------------------------
# 최근 20년 회귀분석
# --------------------------------------------------
end_year = int(yearly["연도"].max())
recent_start_year = end_year - 19

recent20 = yearly[
    yearly["연도"] >= recent_start_year
].copy()

# 최근 20년 데이터가 충분한 경우
if len(recent20) >= 2:

    x_recent = recent20["연도"].to_numpy(dtype=float)
    y_recent = recent20["평균기온"].to_numpy(dtype=float)

    slope_recent, intercept_recent = np.polyfit(
        x_recent,
        y_recent,
        1
    )

    slope_recent_100 = slope_recent * 100

else:
    slope_recent = None
    intercept_recent = None
    slope_recent_100 = None


# --------------------------------------------------
# 제목
# --------------------------------------------------
st.title("🌡️ 기온 예측기")

st.markdown(
    "서울의 **연평균기온 변화 추세**를 이용해 미래의 예상 기온을 계산합니다."
)

st.caption(
    "2025년까지의 자료 중 연간 관측일수가 300일 이상인 연도만 분석에 사용합니다."
)


# --------------------------------------------------
# 100년당 기온 변화 비교
# --------------------------------------------------
st.markdown("---")
st.subheader("📈 100년에 기온이 얼마나 변할까?")

col1, col2 = st.columns(2)

with col1:

    if slope_all_100 >= 0:
        sign_text = "오릅니다"
    else:
        sign_text = "내립니다"

    st.markdown("### 전체 기간")
    st.metric(
        "100년당 변화",
        f"{abs(slope_all_100):.2f} °C",
        delta=f"{slope_all_100:+.2f} °C / 100년"
    )

    st.write(
        f"전체 기간에서는 100년에 약 **{abs(slope_all_100):.2f}°C {sign_text}**."
    )

    st.caption(
        f"{int(yearly['연도'].min())}년 ~ {int(yearly['연도'].max())}년"
    )


with col2:

    st.markdown("### 최근 20년")

    if slope_recent_100 is not None:

        if slope_recent_100 >= 0:
            recent_sign = "오릅니다"
        else:
            recent_sign = "내립니다"

        st.metric(
            "100년당 변화",
            f"{abs(slope_recent_100):.2f} °C",
            delta=f"{slope_recent_100:+.2f} °C / 100년"
        )

        st.write(
            f"최근 20년에서는 100년에 약 "
            f"**{abs(slope_recent_100):.2f}°C {recent_sign}**."
        )

        st.caption(
            f"{int(recent20['연도'].min())}년 ~ {int(recent20['연도'].max())}년"
        )

    else:
        st.warning("최근 20년 데이터가 충분하지 않습니다.")


# --------------------------------------------------
# 전체 기간 vs 최근 20년 비교
# --------------------------------------------------
st.markdown("---")
st.subheader("🔎 전체 기간과 최근 20년 비교")

comparison_col1, comparison_col2 = st.columns(2)

with comparison_col1:

    st.markdown("#### 전체 기간 회귀")

    st.markdown(
        f"<h1 style='font-size:42px;'>"
        f"{slope_all_100:+.2f} °C"
        f"</h1>",
        unsafe_allow_html=True
    )

    st.write("100년당 기온 변화량")

    st.caption(
        f"사용 연도: {int(yearly['연도'].min())}~"
        f"{int(yearly['연도'].max())}"
    )


with comparison_col2:

    st.markdown("#### 최근 20년 회귀")

    if slope_recent_100 is not None:

        st.markdown(
            f"<h1 style='font-size:42px;'>"
            f"{slope_recent_100:+.2f} °C"
            f"</h1>",
            unsafe_allow_html=True
        )

        st.write("100년당 기온 변화량")

        st.caption(
            f"사용 연도: {int(recent20['연도'].min())}~"
            f"{int(recent20['연도'].max())}"
        )


# --------------------------------------------------
# 연도 선택
# --------------------------------------------------
st.markdown("---")
st.subheader("📅 예상 연도 선택")

selected_year = st.slider(
    "연도를 움직여 보세요",
    min_value=1900,
    max_value=2100,
    value=2025,
    step=1
)

# 전체 기간 회귀식을 사용한 예상값
predicted_temperature = (
    slope_all * selected_year + intercept_all
)


# --------------------------------------------------
# 예상 기온 크게 표시
# --------------------------------------------------
st.markdown("---")

st.markdown(
    f"""
    <div style="
        text-align:center;
        padding:30px;
        border-radius:20px;
        background-color:#f5f7fa;
        margin-bottom:20px;
    ">
        <div style="font-size:24px; font-weight:600;">
            {selected_year}년 예상 연평균기온
        </div>
        <div style="font-size:64px; font-weight:800; margin-top:10px;">
            {predicted_temperature:.2f} °C
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# 기본 정보
# --------------------------------------------------
info1, info2, info3 = st.columns(3)

with info1:
    st.metric(
        "분석에 사용한 연도",
        f"{len(yearly)}개"
    )

with info2:
    st.metric(
        "분석 시작 연도",
        f"{int(yearly['연도'].min())}년"
    )

with info3:
    st.metric(
        "분석 끝 연도",
        f"{int(yearly['연도'].max())}년"
    )


# --------------------------------------------------
# 상관계수
# --------------------------------------------------
st.markdown("---")

st.subheader("📊 전체 기간 상관계수")

st.markdown(
    f"<h2>{correlation:.3f}</h2>",
    unsafe_allow_html=True
)

st.write(
    "연도와 연평균기온 사이의 선형적인 관계를 나타내는 값입니다."
)


# --------------------------------------------------
# Plotly 그래프
# --------------------------------------------------
st.markdown("---")
st.subheader("📉 연평균기온 산점도와 회귀 직선")

fig = go.Figure()

# 실제 데이터 산점도
fig.add_trace(
    go.Scatter(
        x=yearly["연도"],
        y=yearly["평균기온"],
        mode="markers",
        name="실제 연평균기온",
        text=[
            f"{year}년<br>"
            f"연평균기온: {temp:.2f}°C<br>"
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


# 전체 기간 회귀선
line_x_all = np.linspace(
    x_all.min(),
    x_all.max(),
    300
)

line_y_all = (
    slope_all * line_x_all +
    intercept_all
)

fig.add_trace(
    go.Scatter(
        x=line_x_all,
        y=line_y_all,
        mode="lines",
        name="전체 기간 회귀선",
        line=dict(
            width=3
        )
    )
)


# 최근 20년 회귀선
if slope_recent_100 is not None:

    line_x_recent = np.linspace(
        x_recent.min(),
        x_recent.max(),
        100
    )

    line_y_recent = (
        slope_recent * line_x_recent +
        intercept_recent
    )

    fig.add_trace(
        go.Scatter(
            x=line_x_recent,
            y=line_y_recent,
            mode="lines",
            name="최근 20년 회귀선",
            line=dict(
                width=3,
                dash="dash"
            )
        )
    )


# 선택한 연도 예상값
fig.add_trace(
    go.Scatter(
        x=[selected_year],
        y=[predicted_temperature],
        mode="markers",
        name="선택 연도 예상값",
        marker=dict(
            size=16,
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
    title="서울 연도별 연평균기온 변화",
    xaxis_title="연도",
    yaxis_title="연평균기온 (°C)",
    height=600,
    template="plotly_white",
    hovermode="closest",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0
    )
)

fig.update_xaxes(dtick=10)

st.plotly_chart(
    fig,
    use_container_width=True
)


# --------------------------------------------------
# 회귀 정보
# --------------------------------------------------
with st.expander("📋 회귀분석 상세 정보"):

    st.write(
        f"**전체 기간 회귀식:** "
        f"연평균기온 = {slope_all:.6f} × 연도 + {intercept_all:.2f}"
    )

    st.write(
        f"**전체 기간 100년 변화량:** "
        f"{slope_all_100:+.2f} °C"
    )

    if slope_recent_100 is not None:

        st.write(
            f"**최근 20년 회귀식:** "
            f"연평균기온 = {slope_recent:.6f} × 연도 "
            f"+ {intercept_recent:.2f}"
        )

        st.write(
            f"**최근 20년 100년 변화량:** "
            f"{slope_recent_100:+.2f} °C"
        )

    st.write(
        f"**전체 기간 상관계수:** {correlation:.3f}"
    )


# --------------------------------------------------
# 사용 데이터
# --------------------------------------------------
with st.expander("📋 회귀에 사용된 연도별 데이터"):

    display_df = yearly.copy()

    display_df["평균기온"] = display_df["평균기온"].round(2)
    display_df["회귀기온"] = (
        slope_all * display_df["연도"] +
        intercept_all
    ).round(2)

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )


st.caption(
    "※ 1900~2100년의 예상값은 전체 분석 기간의 선형 회귀식을 "
    "1900~2100년까지 연장하여 계산한 값입니다."
)
