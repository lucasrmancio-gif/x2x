import matplotlib.pyplot as plt
import streamlit as st
from main import *

TEAMS = [
    "Cruzeiro",
    "Flamengo",
    "Palmeiras",
    "Corinthians",
    "São Paulo",
    "Santos",
    "Grêmio",
    "Internacional",
    "Atlético Mineiro",
    "Botafogo",
    "Fluminense",
    "Vasco",
    "Bahia",
    "Athletico Paranaense",
    "Bragantino",
    "Vitória"
]


st.set_page_config(
    page_title="x2x",
    layout="wide"
)

st.title("x2x")

st.markdown("""
<style>
.card {
    background-color: #1a1f2b;
    padding: clamp(12px, 2vw, 20px);
    border-radius: 15px;
    text-align: center;
    border: 1px solid #2f3747;
    overflow-wrap: break-word;
    word-break: break-word;
}

.card-title {
    font-size: clamp(14px, 1.5vw, 18px);
}

.card-value {
    font-size: clamp(22px, 3vw, 32px);
}
</style>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    home_team = st.selectbox(
        "Mandante",
        TEAMS,
        index=0
    )

with col2:
    away_team = st.selectbox(
        "Visitante",
        TEAMS,
        index=1
    )

if home_team == away_team:
    st.warning("Escolha dois times diferentes.")
    st.stop()

if st.button("Gerar análise"):

    home_id, home_name, home_gols_feitos, home_gols_sofridos = get_last_matches_stats(
        home_team,
        last=10,
        mode="home"
    )

    away_id, away_name, away_gols_feitos, away_gols_sofridos = get_last_matches_stats(
        away_team,
        last=10,
        mode="away"
    )

    league_home_avg, league_away_avg = get_league_average_goals()

    home_attack_strength = home_gols_feitos / league_home_avg
    home_defense_strength = home_gols_sofridos / league_away_avg

    away_attack_strength = away_gols_feitos / league_away_avg
    away_defense_strength = away_gols_sofridos / league_home_avg

    lambda_home = (
        home_attack_strength *
        away_defense_strength *
        league_home_avg
    )

    lambda_away = (
        away_attack_strength *
        home_defense_strength *
        league_away_avg
    )

    lambda_home *= HOME_ADVANTAGE

    matrix = generate_probability_matrix(
        lambda_home,
        lambda_away
    )

    matrix = apply_dixon_coles_adjustment(
        matrix,
        DIXON_COLES_RHO
    )

    home_win_prob, draw_prob, away_win_prob = (
        calculate_match_outcome_probabilities(matrix)
    )

    home_goals, away_goals, score_prob = (
        get_most_probable_score(matrix)
    )

    btts, over_25, under_25 = calculate_markets(matrix)

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="card">
            <div class="card-title">Mandante</div>
            <div class="card-value">{home_win_prob:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="card">
            <div class="card-title">Empate</div>
            <div class="card-value">{draw_prob:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="card">
            <div class="card-title">Visitante</div>
            <div class="card-value">{away_win_prob:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="card">
            <div class="card-title">Ambos Marcam</div>
            <div class="card-value">{btts:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="card">
            <div class="card-title">Over 2.5</div>
            <div class="card-value">{over_25:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="card">
            <div class="card-title">Under 2.5</div>
            <div class="card-value">{under_25:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()

    fig = plot_heatmap(
    matrix,
    home_name,
    away_name,
    home_win_prob,
    draw_prob,
    away_win_prob
    )

    col_left, col_center, col_right = st.columns([1, 2, 1])

    with col_center:
        st.pyplot(fig, use_container_width=True)
