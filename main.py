import requests
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patheffects as path_effects

from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
from scipy.stats import poisson

import streamlit as st

API_KEY = st.secrets["API_KEY"]

BASE_URL = "https://api.football-data.org/v4"
COMPETITION = "BSA"  # Brasileirão Série A

HOME_ADVANTAGE = 1.10
DIXON_COLES_RHO = 0.10

TEAM_LOGOS = {
    "Cruzeiro EC": "logos/Cruzeiro.png",
    "CR Flamengo": "logos/Flamengo.png",
    "SE Palmeiras": "logos/Palmeiras.png",
    "São Paulo FC": "logos/São Paulo.png",
    "Santos FC": "logos/Santos.png",
    "SC Corinthians Paulista": "logos/Corinthians.png",
    "Grêmio FBPA": "logos/Grêmio.png",
    "SC Internacional": "logos/Internacional.png",
    "Atlético Mineiro": "logos/Atlético Mineiro.png",
    "Botafogo FR": "logos/Botafogo.png",
    "Fluminense FC": "logos/Fluminense.png",
    "CR Vasco da Gama": "logos/Vasco da Gama.png",
    "EC Bahia": "logos/Bahia.png",
    "Athletico Paranaense": "logos/Athletico Paranaense.png",
    "Coritiba FBC": "logos/Coritiba.png",
    "Chapecoense AF": "logos/Chapecoense.png",
    "EC Vitória": "logos/Vitória.png",
    "Clube do Remo": "logos/Remo.png",
    "Red Bull Bragantino": "logos/Red Bull Bragantino.png"
}


def get_headers():
    return {"X-Auth-Token": API_KEY}


def get_matches():
    url = f"{BASE_URL}/competitions/{COMPETITION}/matches"
    response = requests.get(url, headers=get_headers())
    data = response.json()

    if "errorCode" in data:
        raise ValueError(data)

    return data["matches"]


def get_team_id(nome_time):
    matches = get_matches()

    for match in matches:
        home = match["homeTeam"]
        away = match["awayTeam"]

        if nome_time.lower() in home["name"].lower():
            return home["id"], home["name"]

        if nome_time.lower() in away["name"].lower():
            return away["id"], away["name"]

    raise ValueError(f"Time não encontrado: {nome_time}")


def get_last_matches_stats(nome_time, last=10, mode="all"):
    team_id, team_name = get_team_id(nome_time)
    matches = get_matches()

    finished_matches = []

    for match in matches:
        if match["status"] != "FINISHED":
            continue

        home = match["homeTeam"]
        away = match["awayTeam"]

        is_home = home["id"] == team_id
        is_away = away["id"] == team_id

        if not (is_home or is_away):
            continue

        if mode == "home" and not is_home:
            continue

        if mode == "away" and not is_away:
            continue

        finished_matches.append(match)

    if not finished_matches:
        raise ValueError(f"Nenhum jogo encontrado para {team_name} no modo {mode}.")

    last_matches = finished_matches[-last:]

    gols_feitos = 0
    gols_sofridos = 0
    peso_total = 0

    for index, match in enumerate(last_matches):
        weight = 1 + (index / len(last_matches))

        home = match["homeTeam"]
        away = match["awayTeam"]

        home_goals = match["score"]["fullTime"]["home"]
        away_goals = match["score"]["fullTime"]["away"]

        if home["id"] == team_id:
            gols_feitos += home_goals * weight
            gols_sofridos += away_goals * weight
        else:
            gols_feitos += away_goals * weight
            gols_sofridos += home_goals * weight

        peso_total += weight

    return (
        team_id,
        team_name,
        gols_feitos / peso_total,
        gols_sofridos / peso_total
    )


def get_league_average_goals():
    matches = get_matches()

    total_home_goals = 0
    total_away_goals = 0
    total_matches = 0

    for match in matches:

        if match["status"] != "FINISHED":
            continue

        home_goals = match["score"]["fullTime"]["home"]
        away_goals = match["score"]["fullTime"]["away"]

        total_home_goals += home_goals
        total_away_goals += away_goals

        total_matches += 1

    avg_home_goals = total_home_goals / total_matches
    avg_away_goals = total_away_goals / total_matches

    return avg_home_goals, avg_away_goals


def generate_probability_matrix(home_lambda, away_lambda, max_goals=5):
    matrix = np.zeros((max_goals + 1, max_goals + 1))

    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            home_prob = poisson.pmf(i, home_lambda)
            away_prob = poisson.pmf(j, away_lambda)
            matrix[i][j] = home_prob * away_prob

    return matrix


def apply_dixon_coles_adjustment(matrix, rho=0.10):
    adjusted_matrix = matrix.copy()

    adjusted_matrix[0][0] *= (1 + rho)
    adjusted_matrix[1][1] *= (1 + rho)

    adjusted_matrix[1][0] *= (1 - rho / 2)
    adjusted_matrix[0][1] *= (1 - rho / 2)

    adjusted_matrix = adjusted_matrix / adjusted_matrix.sum()

    return adjusted_matrix


def calculate_match_outcome_probabilities(matrix):

    home_win = 0
    draw = 0
    away_win = 0

    rows, cols = matrix.shape

    for i in range(rows):
        for j in range(cols):

            if i > j:
                home_win += matrix[i][j]

            elif i == j:
                draw += matrix[i][j]

            else:
                away_win += matrix[i][j]

    return (
        home_win * 100,
        draw * 100,
        away_win * 100
    )


def get_most_probable_score(matrix):

    max_index = np.unravel_index(
        np.argmax(matrix),
        matrix.shape
    )

    home_goals = max_index[0]
    away_goals = max_index[1]

    probability = matrix[max_index] * 100

    return home_goals, away_goals, probability


def calculate_markets(matrix):
    both_teams_score = 0
    over_25 = 0
    under_25 = 0

    rows, cols = matrix.shape

    for home_goals in range(rows):
        for away_goals in range(cols):
            prob = matrix[home_goals][away_goals]

            if home_goals > 0 and away_goals > 0:
                both_teams_score += prob

            if home_goals + away_goals > 2:
                over_25 += prob
            else:
                under_25 += prob

    return both_teams_score * 100, over_25 * 100, under_25 * 100


def add_team_logo(ax, logo_path, xy, zoom=0.12, rotation=0):

    img = Image.open(logo_path).convert("RGBA")

    # rotaciona a imagem
    if rotation != 0:
        img = img.rotate(
            rotation,
            expand=True,
            resample=Image.Resampling.BICUBIC
            )

    imagebox = OffsetImage(img, zoom=zoom, interpolation="hanning")

    ab = AnnotationBbox(
        imagebox,
        xy,
        frameon=False,
        xycoords='axes fraction'
    )

    ax.add_artist(ab)


def plot_heatmap(matrix, home_team, away_team, home_win, draw, away_win):

    matrix_percent = matrix * 100

    max_index = np.unravel_index(np.argmax(matrix), matrix.shape)
    most_home_goals = max_index[0]
    most_away_goals = max_index[1]
    most_prob = matrix[max_index] * 100

    labels = np.array([
        [f"{value:.1f}%" for value in row]
        for row in matrix_percent
    ])

    plt.figure(figsize=(5, 4), facecolor="#0f1419")

    ax = sns.heatmap(
        matrix_percent,
        annot=labels,
        fmt="",
        cmap="crest",
        cbar=False, # remove barra lateral
        linewidths=3,
        linecolor="#0f1419",
        square=True
    )

    ax.set_facecolor("#0f1419")

    ax.tick_params(colors="white", labelsize=8)

    for text in ax.texts:
        text.set_color("white")
        text.set_fontsize(6)
        text.set_fontweight("bold")

        text.set_path_effects([
            path_effects.Stroke(linewidth=0.5, foreground="black"),
            path_effects.Normal()
        ])

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    home_logo = TEAM_LOGOS.get(home_team)
    away_logo = TEAM_LOGOS.get(away_team)

    # Logo do mandante
    if home_logo:
        add_team_logo(
            ax,
            home_logo,
            (-0.075, 0.50),
            zoom=0.018
        )

    # Logo do visitante
    if away_logo:
        add_team_logo(
            ax,
            away_logo,
            (0.50, -0.055),
            zoom=0.018
        )
    return plt.gcf()


if __name__ == "__main__":

    home_team = "Cruzeiro"
    away_team = "Flamengo"

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

    # MÉDIA DA LIGA
    league_home_avg, league_away_avg = get_league_average_goals()

    # FORÇAS
    home_attack_strength = home_gols_feitos / league_home_avg
    home_defense_strength = home_gols_sofridos / league_away_avg

    away_attack_strength = away_gols_feitos / league_away_avg
    away_defense_strength = away_gols_sofridos / league_home_avg

    # LAMBDAS
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

    matrix = generate_probability_matrix(lambda_home, lambda_away)
    matrix = apply_dixon_coles_adjustment(matrix, DIXON_COLES_RHO)

    home_win_prob, draw_prob, away_win_prob = calculate_match_outcome_probabilities(matrix)

    print(f"Probabilidade de vitória do mandante: {home_win_prob:.2f}%")
    print(f"Probabilidade de empate: {draw_prob:.2f}%")
    print(f"Probabilidade de vitória do visitante: {away_win_prob:.2f}%")

    home_goals, away_goals, score_prob = get_most_probable_score(matrix)

    print(
        f"Placar mais provável: "
        f"{home_goals} x {away_goals} "
        f"({score_prob:.2f}%)"
    )

    plot_heatmap(matrix, home_name, away_name, home_win_prob, draw_prob, away_win_prob)

    btts, over_25, under_25 = calculate_markets(matrix)

    print(f"Ambas marcam: {btts:.2f}%")
    print(f"Over 2.5 gols: {over_25:.2f}%")
    print(f"Under 2.5 gols: {under_25:.2f}%")