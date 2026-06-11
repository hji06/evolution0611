import streamlit as st
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="진화 가상 실험실", layout="wide")

st.title("🧬 진화 가상 실험실: 가설과 검증")
st.caption("통합과학(자연선택)과 생명과학2(하디-바인베르크 법칙) 융합 탐구 시뮬레이터")

# --- 1. 사이드바: 시나리오 및 가설 설정 ---
with st.sidebar:
    st.header("🔬 탐구 주제 선택")
    scenario = st.selectbox(
        "어떤 진화 요인을 연구할까요?",
        ["👃 시나리오 1: 성선택 (코 모양)", 
         "☀️ 시나리오 2: 온도 적응 I (피부색)", 
         "❄️ 시나리오 3: 온도 적응 II (털 밀도)"]
    )
    
    st.markdown("---")
    st.header("🎛️ 독립변인 설정 (환경 조작)")
    
    # 시나리오별 동적 UI 및 적합도(w) 계산 로직
    if "성선택" in scenario:
        st.info("A: 일반 코 (우성) / a: 돼지코 (열성)")
        mating_success = st.slider("돼지코(aa)의 짝짓기 성공률", 0.0, 1.0, 0.4, step=0.1)
        w_AA = 1.0
        w_Aa = 1.0
        w_aa = mating_success
        hypothesis_text = f"돼지코의 짝짓기 성공률이 {mating_success*100:.0f}% 로 낮아진다면, a 유전자 빈도는 감소할 것이다."
        
    elif "피부색" in scenario:
        st.info("A: 어두운 피부 (우성) / a: 밝은 피부 (열성)\n\n*어두운 피부는 더위에 강합니다.")
        temp = st.slider("환경 온도 (°C)", 10, 50, 30, step=1)
        # 온도(10~50)를 0.0~1.0 비율로 변환
        t_norm = (temp - 10) / 40.0 
        w_AA = 0.4 + (0.6 * t_norm)  # 고온일수록 생존율 1.0에 가까워짐
        w_Aa = w_AA                  # 완전 우성
        w_aa = 1.0 - (0.6 * t_norm)  # 고온일수록 생존율 하락
        hypothesis_text = f"환경 온도가 {temp}°C 일 때, 더위에 강한 어두운 피부(A)의 빈도는 어떻게 변할까?"
        
    else: # 털 밀도
        st.info("A: 털 많음 / a: 털 없음 / Aa: 중간 (불완전 우성)\n\n*털이 많으면 추위에 강하고 더위에 약합니다.")
        temp = st.slider("환경 온도 (°C)", -30, 40, 5, step=1)
        # 온도(-30~40)를 0.0~1.0 비율로 변환
        t_norm = (temp + 30) / 70.0
        w_AA = 1.0 - (0.9 * t_norm)  # 고온(t_norm=1)에서 생존율 0.1로 급감
        w_aa = 0.1 + (0.9 * t_norm)  # 고온에서 생존율 1.0으로 증가
        w_Aa = (w_AA + w_aa) / 2     # 이형접합자는 중간 생존율
        hypothesis_text = f"환경 온도가 {temp}°C 인 지역에서, 털이 많은 개체와 적은 개체 중 누가 살아남을까?"

    st.markdown("---")
    st.header("🧬 통제변인 설정")
    p_init = st.slider("초기 대립유전자 A 빈도 (p)", 0.0, 1.0, 0.5, step=0.01)
    
    use_drift = st.checkbox("유전적 부동 활성화 (제한된 집단)")
    pop_size = st.number_input("집단 크기 (N)", min_value=10, max_value=5000, value=100, step=10, disabled=not use_drift)
    
    mu = st.slider("돌연변이율 (A↔a)", 0.000, 0.050, 0.000, step=0.001, format="%.3f")
    
    if st.button("🔄 결과 새로고침", type="primary"):
        pass

# --- 2. 메인 화면: 가설 및 실험 환경 ---
st.success(f"**📝 나의 실험 가설:** {hypothesis_text}")

# 유전자형 생존가 시각적 확인
col_w1, col_w2, col_w3 = st.columns(3)
col_w1.metric("AA 유전자형 생존가", f"{w_AA:.2f}")
col_w2.metric("Aa 유전자형 생존가", f"{w_Aa:.2f}")
col_w3.metric("aa 유전자형 생존가", f"{w_aa:.2f}")

# --- 3. 시뮬레이션 알고리즘 로직 ---
MAX_GENERATIONS = 200

@st.cache_data
def run_simulation(p_init, w_AA, w_Aa, w_aa, mu, use_drift, pop_size, max_gen, _random_trigger):
    p_hist = [p_init]
    q_hist = [1.0 - p_init]
    curr_p = p_init
    
    for _ in range(max_gen):
        curr_q = 1.0 - curr_p
        
        # 1. 자연선택
        w_bar = (curr_p**2) * w_AA + (2 * curr_p * curr_q) * w_Aa + (curr_q**2) * w_aa
        if w_bar > 0:
            p_sel = ((curr_p**2) * w_AA + curr_p * curr_q * w_Aa) / w_bar
        else:
            p_sel = 0.0
            
        # 2. 돌연변이 (단순화를 위해 양방향 동일 확률 적용)
        p_mut = p_sel * (1 - mu) + (1 - p_sel) * mu
        
        # 3. 유전적 부동
        if use_drift and pop_size > 0:
            success_count = np.random.binomial(2 * pop_size, p_mut)
            next_p = success_count / (2 * pop_size)
        else:
            next_p = p_mut
            
        p_hist.append(next_p)
        q_hist.append(1.0 - next_p)
        curr_p = next_p
        
    return p_hist, q_hist

random_trigger = np.random.rand() if use_drift else 0
p_history, q_history = run_simulation(p_init, w_AA, w_Aa, w_aa, mu, use_drift, pop_size, MAX_GENERATIONS, random_trigger)

# --- 4. 세대 탐구 슬라이더 ---
st.markdown("---")
st.subheader("⏱️ 세대 흐름 관찰")
target_gen = st.slider("관찰할 세대", 0, MAX_GENERATIONS, 100)

view_p = p_history[:target_gen+1]
view_q = q_history[:target_gen+1]
current_p = view_p[-1]
current_q = view_q[-1]

# --- 5. 결과 시각화 (유전자 빈도 및 표현형 빈도) ---
fig = go.Figure()
generations = list(range(target_gen + 1))

fig.add_trace(go.Scatter(x=generations, y=view_p, mode='lines', name='p (A 빈도)', line=dict(color='#1f77b4', width=4)))
fig.add_trace(go.Scatter(x=generations, y=view_q, mode='lines', name='q (a 빈도)', line=dict(color='#ff7f0e', width=4)))

fig.update_layout(
    title="대립유전자 빈도 변화",
    xaxis_title="세대 (Generation)", 
    yaxis_title="빈도", 
    yaxis=dict(range=[0, 1.05]), 
    xaxis=dict(range=[0, MAX_GENERATIONS]), 
    margin=dict(l=40, r=40, t=40, b=40), 
    hovermode="x unified"
)
st.plotly_chart(fig, use_container_width=True)

# 표현형(유전자형) 빈도 분석
st.subheader(f"📊 {target_gen}세대 표현형(유전자형) 비율")
p_sq = current_p**2
two_pq = 2 * current_p * current_q
q_sq = current_q**2

c1, c2, c3 = st.columns(3)
if "털 밀도" in scenario:
    c1.metric("털 많음 (AA)", f"{p_sq*100:.1f}%")
    c2.metric("중간 털 (Aa)", f"{two_pq*100:.1f}%")
    c3.metric("털 없음 (aa)", f"{q_sq*100:.1f}%")
else:
    c1.metric("우성 표현형 (AA)", f"{p_sq*100:.1f}%")
    c2.metric("우성 표현형 (Aa)", f"{two_pq*100:.1f}%", help="보인자 역할")
    c3.metric("열성 표현형 (aa)", f"{q_sq*100:.1f}%")

st.progress(p_sq + two_pq + q_sq) # 1.0(100%) 비율 바
