import streamlit as st
import numpy as np
import plotly.graph_objects as go

# 페이지 설정
st.set_page_config(page_title="진화 시뮬레이터", layout="wide")

st.title("🧬 하디-바인베르크 평형 및 진화 시뮬레이터")
st.caption("자연선택, 돌연변이, 유전적 부동을 탐구하는 웹앱")

# --- 1. 사이드바: 실험 조건 설정 ---
with st.sidebar:
    st.header("🎛️ 실험 조건 설정")
    
    p_init = st.slider("초기 대립유전자 A 빈도 (p)", 0.0, 1.0, 0.5, step=0.01)
    
    st.markdown("---")
    st.subheader("1. 자연선택 (상대 적합도)")
    st.caption("각 유전자형의 생존 능력을 0.0 ~ 1.0 사이로 설정하세요.")
    w_AA = st.slider("AA 유전자형 (w_AA)", 0.0, 1.0, 1.0, step=0.1)
    w_Aa = st.slider("Aa 유전자형 (w_Aa)", 0.0, 1.0, 1.0, step=0.1)
    w_aa = st.slider("aa 유전자형 (w_aa)", 0.0, 1.0, 1.0, step=0.1)
    
    st.markdown("---")
    st.subheader("2. 돌연변이")
    st.caption("한 대립유전자가 다른 대립유전자로 변할 확률입니다.")
    mu = st.slider("A → a 돌연변이율 (μ)", 0.000, 0.100, 0.000, step=0.001, format="%.3f")
    nu = st.slider("a → A 돌연변이율 (ν)", 0.000, 0.100, 0.000, step=0.001, format="%.3f")
    
    st.markdown("---")
    st.subheader("3. 유전적 부동 (집단 크기)")
    use_drift = st.checkbox("유전적 부동 활성화 (무작위 추출)")
    pop_size = st.number_input("집단 크기 (N)", min_value=10, max_value=10000, value=100, step=10, disabled=not use_drift)
    
    st.markdown("---")
    # 부동이 활성화되어 있을 때 난수 재시도를 위한 버튼
    if st.button("🔄 결과 새로고침", type="primary"):
        pass

# --- 2. 시뮬레이션 알고리즘 로직 ---
# 부드러운 드래그 구현을 위해 200세대까지의 데이터를 미리 연산합니다.
MAX_GENERATIONS = 200

@st.cache_data
def run_simulation(p_init, w_AA, w_Aa, w_aa, mu, nu, use_drift, pop_size, max_gen, _random_trigger):
    """
    파라미터가 바뀔 때만 재연산되도록 설정. 
    _random_trigger는 새로고침 버튼을 눌렀을 때 난수 생성을 다시 하도록 돕는 변수입니다.
    """
    p_hist = [p_init]
    q_hist = [1.0 - p_init]
    
    curr_p = p_init
    
    for _ in range(max_gen):
        curr_q = 1.0 - curr_p
        
        # [1] 자연선택
        w_bar = (curr_p**2) * w_AA + (2 * curr_p * curr_q) * w_Aa + (curr_q**2) * w_aa
        if w_bar > 0:
            p_sel = ((curr_p**2) * w_AA + curr_p * curr_q * w_Aa) / w_bar
        else:
            p_sel = 0.0
            
        # [2] 돌연변이
        p_mut = p_sel * (1 - mu) + (1 - p_sel) * nu
        
        # [3] 유전적 부동
        if use_drift and pop_size > 0:
            success_count = np.random.binomial(2 * pop_size, p_mut)
            next_p = success_count / (2 * pop_size)
        else:
            next_p = p_mut
            
        p_hist.append(next_p)
        q_hist.append(1.0 - next_p)
        curr_p = next_p
        
    return p_hist, q_hist

# 파라미터를 기반으로 전체 역사를 연산
random_trigger = np.random.rand() if use_drift else 0 # 유전적 부동일 때만 매번 난수 갱신
p_history, q_history = run_simulation(p_init, w_AA, w_Aa, w_aa, mu, nu, use_drift, pop_size, MAX_GENERATIONS, random_trigger)

# --- 3. UI: 세대 탐구 슬라이더 ---
st.subheader("⏱️ 세대 탐구")
st.markdown("슬라이더를 좌우로 드래그하여 특정 세대의 상태를 확인하세요.")
# 사용자가 선택한 세대까지만 데이터를 슬라이싱하여 보여줌 (빠른 응답성 보장)
target_gen = st.slider("관찰할 세대", 0, MAX_GENERATIONS, 50)

view_p = p_history[:target_gen+1]
view_q = q_history[:target_gen+1]
current_p = view_p[-1]
current_q = view_q[-1]

# --- 4. 결과 시각화 ---
col1, col2, col3 = st.columns(3)
col1.metric("현재 세대", f"{target_gen} 세대")
col2.metric("대립유전자 A 빈도 (p)", f"{current_p:.4f}")
col3.metric("대립유전자 a 빈도 (q)", f"{current_q:.4f}")

fig = go.Figure()
generations = list(range(target_gen + 1))

fig.add_trace(go.Scatter(x=generations, y=view_p, mode='lines', name='p (A 빈도)', line=dict(color='#1f77b4', width=4)))
fig.add_trace(go.Scatter(x=generations, y=view_q, mode='lines', name='q (a 빈도)', line=dict(color='#ff7f0e', width=4)))

# 그래프 설정 (X축을 MAX_GENERATIONS로 고정하여 드래그 시 축이 요동치는 것을 방지)
fig.update_layout(
    xaxis_title="세대 (Generation)", 
    yaxis_title="대립유전자 빈도", 
    yaxis=dict(range=[0, 1.05]), 
    xaxis=dict(range=[0, MAX_GENERATIONS]), 
    margin=dict(l=40, r=40, t=20, b=40), 
    hovermode="x unified"
)

st.plotly_chart(fig, use_container_width=True)

# 유전자형 빈도 표출 (수학 교과 연계)
st.subheader(f"🧬 {target_gen}세대 유전자형 빈도")
c1, c2, c3 = st.columns(3)
c1.metric("우성 동형접합 (AA, p²)", f"{current_p**2:.4f}")
c2.metric("이형접합 (Aa, 2pq)", f"{2 * current_p * current_q:.4f}")
c3.metric("열성 동형접합 (aa, q²)", f"{current_q**2:.4f}")
