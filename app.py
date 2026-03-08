import streamlit as st
import FinanceDataReader as fdr
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from datetime import datetime, timedelta

# 1. 페이지 테마 설정 (와이드 모드 및 다크 테마 지향)
st.set_page_config(page_title="PRO 주가 분석기", layout="wide")

# 2. 스타일링 (CSS 인젝션으로 디자인 변경)
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #1e2130; border-radius: 10px; padding: 15px; border: 1px solid #3e4255; }
    </style>
    """, unsafe_allow_html=True)

st.title("📊 PRO Stock Intelligence")
st.caption("실시간 데이터 기반 전략 분석 및 AI 예측")

# 3. 사이드바 - 입력창 구성 변경
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2422/2422796.png", width=100)
    st.header("Terminal")
    target = st.text_input("종목 코드", value="005930", help="숫자 6자리를 입력하세요")
    range_select = st.select_slider("데이터 범위", options=['3개월', '6개월', '1년', '3년'], value='1년')
    days_map = {'3개월': 90, '6개월': 180, '1년': 365, '3년': 1095}
    
    predict_days = st.number_input("예측 기간(일)", min_value=5, max_value=60, value=20)
    run_btn = st.button("RUN ANALYSIS", use_container_width=True, type="primary")

if run_btn:
    try:
        # 데이터 로드
        end_d = datetime.now()
        start_d = end_d - timedelta(days=days_map[range_select])
        df = fdr.DataReader(target, start_d, end_d)

        if df.empty:
            st.error("Invalid Ticker")
        else:
            # 보조지표 계산 (이동평균선)
            df['MA20'] = df['Close'].rolling(window=20).mean()
            df['MA60'] = df['Close'].rolling(window=60).mean()

            # AI 예측 (Holt-Winters)
            model = ExponentialSmoothing(df['Close'], trend='add', seasonal=None).fit()
            pred = model.forecast(predict_days)
            pred_dates = [df.index[-1] + timedelta(days=i) for i in range(1, predict_days+1)]

            # --- 레이아웃 배치 ---
            # 상단 지표 (Metric)
            m1, m2, m3, m4 = st.columns(4)
            current_p = df['Close'].iloc[-1]
            prev_p = df['Close'].iloc[-2]
            m1.metric("현재가", f"{current_p:,.0f}원", f"{current_p-prev_p:,.0f}원")
            m2.metric("최고가 (기간내)", f"{df['High'].max():,.0f}원")
            m3.metric("최저가 (기간내)", f"{df['Low'].min():,.0f}원")
            m4.metric("AI 목표가", f"{pred.iloc[-1]:,.0f}원", f"{pred.iloc[-1]-current_p:,.0f}원")

            # 메인 차트 (캔들스틱 + 거래량)
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                               vertical_spacing=0.05, row_heights=[0.7, 0.3])

            # 캔들스틱 차트
            fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'],
                          low=df['Low'], close=df['Close'], name="Price"), row=1, col=1)
            
            # 이동평균선
            fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], name="MA20", line=dict(color='yellow', width=1)), row=1, col=1)
            
            # AI 예측선
            fig.add_trace(go.Scatter(x=pred_dates, y=pred, name="AI Forecast", line=dict(color='#00ff00', width=2, dash='dot')), row=1, col=1)

            # 거래량 차트
            fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name="Volume", marker_color='gray'), row=2, col=1)

            fig.update_layout(height=600, template="plotly_dark", showlegend=False, 
                              xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)

            # 데이터 테이블
            with st.expander("Raw Data 상세 보기"):
                st.dataframe(df.tail(10), use_container_width=True)

    except Exception as e:
        st.warning(f"분석 중 오류 발생: {e}")
else:
    st.info("좌측 터미널에서 분석할 종목을 입력하고 RUN 버튼을 누르세요.")
