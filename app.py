import streamlit as st
import numpy as np
import plotly.graph_objects as go

# 페이지 설정
st.set_page_config(page_title="HWE 시뮬레이터", layout="wide")

st.title("🧬 하디-바인베르크 평형 및 진화 시뮬레이터")
st.caption("생명과학-수학 융합 수업 탐구 활동용 웹앱")

# 1. 세션 상태(상태 관리) 초기화
if 'generation' not in st.session_state:
    st.session_state.generation = 0
if 'p_history' not in st.session_state:
    st.session_state.p_history = []
if 'q_history' not in st.session_state:
    st.session_state.q_history = []

# 2. 사이드바: 조건 설정 (유저 입력)
with st.sidebar:
    st.header("🎛️ 실험 조건 설정")
    
    # 초기 대립유전자 빈도
    p_init = st.slider("초기 대립유전자 A 빈도 (p)", 0.0, 1.0, 0.5, step=0.05)
    q_init = 1.0 - p_init
    st.info(f"초기 대립유전자 a 빈도 (q): {q_init:.2f}")
    
    st.markdown("---")
    st.subheader("⚠️ 평형 붕괴 요인 활성화")
    
    # 유전적 부동 설정
    use_drift = st.checkbox("유전적 부동 (작은 집단 효과) 반영")
    pop_size = st.number_input("집단 크기 (N)", min_value=10, max_value=10000, value=100, step=10, disabled=not use_drift)
    
    # 자연선택 설정
    use_selection = st.checkbox("자연선택 (열성치사 등) 반영")
    w_aa = st.slider("aa 유전자형의 상대 적합도 (w)", 0.0, 1.0, 1.0, step=0.1, disabled=not use_selection)

    st.markdown("---")
    # 리셋 버튼
    if st.button("🔄 시뮬레이션 초기화", type="primary"):
        st.session_state.generation = 0
        st.session_state.p_history = [p_init]
        st.session_state.q_history = [q_init]
        st.rerun()

# 최초 실행 시 초기값 주입
if st.session_state.generation == 0 and len(st.session_state.p_history) == 0:
    st.session_state.p_history.append(p_init)
    st.session_state.q_history.append(q_init)

# 3. 메인 화면: 시뮬레이션 실행 컨트롤
col1, col2 = st.columns(2)
with col1:
    if st.button("🚀 +1세대 진행"):
        current_p = st.session_state.p_history[-1]
        current_q = 1.0 - current_p
        
        # [알고리즘] 1. 자연선택 계산
        if use_selection:
            # AA=1.0, Aa=1.0, aa=w_aa 가정
            w_bar = (current_p**2) * 1.0 + (2 * current_p * current_q) * 1.0 + (current_q**2) * w_aa
            if w_bar > 0:
                p_expected = ((current_p**2) * 1.0 + current_p * current_q * 1.0) / w_bar
            else:
                p_expected = 0
        else:
            p_expected = current_p
            
        # [알고리즘] 2. 유전적 부동 계산 (이항분포 샘플링)
        if use_drift and pop_size > 0:
            # 2N개의 대립유전자 중 A가 무작위 추출될 확률
            success_count = np.random.binomial(2 * pop_size, p_expected)
            next_p = success_count / (2 * pop_size)
        else:
            next_p = p_expected
            
        # 상태 업데이트
        st.session_state.p_history.append(next_p)
        st.session_state.q_history.append(1.0 - next_p)
        st.session_state.generation += 1
        st.rerun()

with col2:
    st.metric(label="현재 진행 세대", value=f"{st.session_state.generation} 세대")

# 4. 결과 시각화 (Plotly 꺾은선 그래프)
st.subheader("📈 세대별 대립유전자 빈도 변화 추이")

fig = go.Figure()
generations = list(range(len(st.session_state.p_history)))

fig.add_trace(go.Scatter(x=generations, y=st.session_state.p_history, mode='lines+markers', name='p (A 빈도)', line=dict(color='#1f77b4', width=3)))
fig.add_trace(go.Scatter(x=generations, y=st.session_state.q_history, mode='lines+markers', name='q (a 빈도)', line=dict(color='#ff7f0e', width=3)))

fig.update_layout(xaxis_title="세대 (Generation)", yaxis_title="빈도 (Frequency)", yaxis=dict(range=[0, 1.05]), margin=dict(l=40, r=40, t=20, b=40), hovermode="x unified")

st.plotly_chart(fig, use_container_width=True)

# 5. 유전자형 빈도 표출 (수학 교과 연계)
p_curr = st.session_state.p_history[-1]
q_curr = st.session_state.q_history[-1]

st.subheader("🧬 현재 세대의 이론적 유전자형 빈도 (HWE 수식)")
c1, c2, c3 = st.columns(3)
c1.metric("Homogenous Dominant (AA, p²)", f"{p_curr**2:.4f}")
c2.metric("Heterogenous (Aa, 2pq)", f"{2*p_curr*q_curr:.4f}")
c3.metric("Homogenous Recessive (aa, q²)", f"{q_curr**2:.4f}")
