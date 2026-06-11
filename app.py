import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd

# ---------------------------------------------------------
# 1. 시뮬레이션 엔진 (Model)
# ---------------------------------------------------------
class EvolutionSimulator:
    def __init__(self, p_init, N, w_AA, w_Aa, w_aa, mu=0.0):
        self.p_init = p_init
        self.N = N
        self.w_AA = w_AA
        self.w_Aa = w_Aa
        self.w_aa = w_aa
        self.mu = mu
        
    def run(self, generations):
        records = []
        curr_p = self.p_init
        
        # HWE 이론값은 변하지 않음
        hw_p = self.p_init
        hw_q = 1.0 - hw_p
        hw_AA = hw_p**2
        hw_Aa = 2 * hw_p * hw_q
        hw_aa = hw_q**2

        for gen in range(generations + 1):
            curr_q = 1.0 - curr_p
            
            # 실제 유전자형 빈도 (현재 대립유전자 풀 기준)
            # 엄밀한 개체 단위 시뮬레이션 대신 수학적 기댓값을 이용한 부동 시뮬레이션
            actual_AA = curr_p**2
            actual_Aa = 2 * curr_p * curr_q
            actual_aa = curr_q**2
            
            # 오차 계산
            error_p = abs(hw_p - curr_p)
            
            # 데이터 기록
            records.append({
                'Generation': gen,
                'HW_p': hw_p, 'HW_q': hw_q,
                'HW_AA': hw_AA, 'HW_Aa': hw_Aa, 'HW_aa': hw_aa,
                'Evo_p': curr_p, 'Evo_q': curr_q,
                'Evo_AA': actual_AA, 'Evo_Aa': actual_Aa, 'Evo_aa': actual_aa,
                'Error_p': error_p
            })
            
            # --- 다음 세대 연산 (진화 요인 적용) ---
            # 1. 자연선택
            w_bar = (curr_p**2) * self.w_AA + (2 * curr_p * curr_q) * self.w_Aa + (curr_q**2) * self.w_aa
            if w_bar > 0:
                p_sel = ((curr_p**2) * self.w_AA + curr_p * curr_q * self.w_Aa) / w_bar
            else:
                p_sel = 0.0
                
            # 2. 돌연변이 (A -> a 단방향 예시)
            p_mut = p_sel * (1 - self.mu)
            
            # 3. 유전적 부동 (이항분포 샘플링)
            if self.N > 0:
                successes = np.random.binomial(2 * self.N, p_mut)
                next_p = successes / (2 * self.N)
            else:
                next_p = p_mut
                
            curr_p = next_p
            
        return pd.DataFrame(records)

# ---------------------------------------------------------
# 2. Streamlit UI (View & Controller)
# ---------------------------------------------------------
st.set_page_config(page_title="진화 가상 실험실", layout="wide")

st.title("🧬 진화 가상 실험실: 모델 빌더")
st.markdown("수학적 이상 모델(HWE)과 실제 진화 현상의 차이를 탐구합니다.")

# --- 사이드바: 변인 통제 및 가설 설정 ---
with st.sidebar:
    st.header("1. 과학적 가설 설정")
    hypothesis = st.text_area("탐구할 가설을 작성하세요.", placeholder="예: 집단 크기(N)가 작을수록 유전적 부동에 의해 이론값(HWE)과의 오차가 커질 것이다.")
    
    st.header("2. 변인 설계 (Model Builder)")
    p_init = st.slider("초기 A 대립유전자 빈도 (p)", 0.0, 1.0, 0.5, 0.01)
    
    st.subheader("진화 요인 설정")
    N = st.number_input("집단 크기 (N)", min_value=10, max_value=10000, value=100, step=10)
    w_AA = st.slider("AA 생존가 (w)", 0.0, 1.0, 1.0, 0.1)
    w_Aa = st.slider("Aa 생존가 (w)", 0.0, 1.0, 1.0, 0.1)
    w_aa = st.slider("aa 생존가 (w)", 0.0, 1.0, 1.0, 0.1)
    mu = st.slider("돌연변이율 (A→a)", 0.0, 0.05, 0.0, 0.001)

# --- 메인 화면: 가설 노출 및 시뮬레이션 ---
if hypothesis:
    st.info(f"**📌 나의 가설:** {hypothesis}")
else:
    st.warning("사이드바에 탐구할 가설을 먼저 작성해주세요.")

target_gen = st.slider("관찰할 세대 수", 10, 500, 100)

# 시뮬레이터 인스턴스 생성 및 실행
simulator = EvolutionSimulator(p_init, N, w_AA, w_Aa, w_aa, mu)
df = simulator.run(target_gen)

# --- 결과 분석 탭 ---
tab1, tab2 = st.tabs(["📊 대립유전자 빈도 비교 (이론 vs 실제)", "📉 오차 분석 (HWE 모델의 붕괴)"])

with tab1:
    fig1 = go.Figure()
    # HWE 이론 모델 (대조군 - 점선)
    fig1.add_trace(go.Scatter(x=df['Generation'], y=df['HW_p'], mode='lines', name='이론적 p (HWE)', line=dict(dash='dash', color='gray')))
    # 실제 진화 모델 (실험군 - 실선)
    fig1.add_trace(go.Scatter(x=df['Generation'], y=df['Evo_p'], mode='lines', name='실제 p (진화)', line=dict(color='blue')))
    
    fig1.update_layout(title="이론과 실제의 대립유전자 빈도 차이", xaxis_title="세대", yaxis_title="대립유전자 빈도 p", yaxis_range=[0, 1])
    st.plotly_chart(fig1, use_container_width=True)

with tab2:
    fig2 = go.Figure()
    # 오차 그래프
    fig2.add_trace(go.Scatter(x=df['Generation'], y=df['Error_p'], mode='lines', fill='tozeroy', name='|이론 p - 실제 p|', line=dict(color='red')))
    
    fig2.update_layout(title="HWE 모델과 실제 시뮬레이션 간의 오차율", xaxis_title="세대", yaxis_title="오차 (Error)", yaxis_range=[0, 1])
    st.plotly_chart(fig2, use_container_width=True)
    
    st.markdown("💡 **탐구 질문:** 그래프의 오차(Error)가 0에서 벗어나 요동치기 시작하는 이유는 무엇일까요? 본인이 설정한 조작 변인과 연결하여 설명해 보세요.")
