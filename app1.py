import streamlit as st
import pandas as pd
import plotly.express as px
import yfinance as yf
from sklearn.linear_model import LinearRegression
import numpy as np
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import os
from textblob import TextBlob
import time
import random
import io
from io import BytesIO
import json
import qrcode
import matplotlib.pyplot as plt
from newsapi import NewsApiClient
from prophet import Prophet
from datetime import datetime, timedelta
import plotly.io as pio
from scipy.optimize import minimize
import tempfile

# Configuration de la page
st.set_page_config(page_title="Tableau de Bord Financier", layout="wide")

# Fonctions financières de base
def calcul_interets_composes(principal, taux, annees):
    return principal * (1 + taux / 100) ** annees

def calcul_amortissement_pret(principal, taux_annuel, annees):
    taux_mensuel = taux_annuel / 12 / 100
    nombre_paiements = annees * 12
    paiement_mensuel = principal * (taux_mensuel * (1 + taux_mensuel) ** nombre_paiements) / ((1 + taux_mensuel) ** nombre_paiements - 1)
    return paiement_mensuel

def calcul_van(flux, taux):
    return sum(f / (1 + taux / 100) ** (t + 1) for t, f in enumerate(flux))

def calcul_tri(flux):
    def van_at_rate(rate):
        return sum(f / (1 + rate) ** (t + 1) for t, f in enumerate(flux))
    low, high = -0.99, 100.0
    for _ in range(100):
        mid = (low + high) / 2
        van = van_at_rate(mid)
        if abs(van) < 0.01:
            return mid * 100
        elif van > 0:
            low = mid
        else:
            high = mid
    return mid * 100

@st.cache_data
def get_stock_data(ticker, period="1y"):
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    return df

def predict_stock_price(df):
    df['Days'] = np.arange(len(df))
    X = df[['Days']]
    y = df['Close']
    model = LinearRegression()
    model.fit(X, y)
    future_days = np.arange(len(df), len(df) + 30).reshape(-1, 1)
    predictions = model.predict(future_days)
    return predictions

def monte_carlo_simulation(initial_investment, mean_return, volatility, years, simulations=1000):
    total_days = 252 * years
    daily_returns = np.random.normal(mean_return / 252, volatility / np.sqrt(252), (total_days, simulations))
    price_paths = initial_investment * np.exp(np.cumsum(daily_returns, axis=0))
    return price_paths

def get_stock_params(ticker, period="1y"):
    df = get_stock_data(ticker, period)
    daily_returns = df['Close'].pct_change().dropna()
    mean_return = daily_returns.mean() * 252
    volatility = daily_returns.std() * np.sqrt(252)
    return mean_return, volatility

def generate_enriched_text(prompt, data, key):
    base_text = f"{key} : {data:.2f}"
    blob = TextBlob(prompt)
    sentiment = blob.sentiment.polarity
    if sentiment > 0:
        return f"{base_text}. Cette valeur reflète une tendance positive, suggérant une opportunité favorable."
    elif sentiment < 0:
        return f"{base_text}. Cette valeur indique une situation préoccupante qui mérite une attention particulière."
    else:
        return f"{base_text}. Les données sont neutres, sans tendance marquée."

def analyze_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

def generate_pdf_report_with_graphics(user_info, selected_data, graphs=None, filename="rapport_personnalise.pdf"):
    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter
    y = height - 50  # Position verticale initiale

    # En-tête personnalisé avec les données de l'utilisateur
    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, y, f"Rapport Financier Personnalisé - {user_info['prenom']} {user_info['nom']}")
    y -= 20
    c.setFont("Helvetica", 12)
    c.drawString(100, y, f"Âge : {user_info['age']} | Email : {user_info['email']}")
    y -= 30

    # Contenu textuel
    c.setFont("Helvetica", 10)
    for key, value in selected_data.items():
        if y < 50:
            c.showPage()
            y = height - 50
            c.setFont("Helvetica", 10)
        c.drawString(100, y, f"{key} :")
        y -= 15
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                if y < 50:
                    c.showPage()
                    y = height - 50
                    c.setFont("Helvetica", 10)
                if isinstance(sub_value, pd.DataFrame):
                    c.drawString(120, y, f"{sub_key} : Tableaux inclus ci-dessous")
                    y -= 15
                else:
                    c.drawString(120, y, f"- {sub_key} : {sub_value}")
                    y -= 15
        else:
            if y < 50:
                c.showPage()
                y = height - 50
                c.setFont("Helvetica", 10)
            c.drawString(120, y, str(value))
            y -= 15

    # Ajout des graphiques et matrices
    if graphs:
        for graph_name, content in graphs.items():
            if y < 300:  # Réserve de l'espace pour le graphique ou la matrice
                c.showPage()
                y = height - 50
            c.drawString(100, y, f"{graph_name} :")
            y -= 20
            if isinstance(content, pd.DataFrame):  # Si c'est une matrice ou un tableau
                # Convertir DataFrame en texte pour PDF
                for idx, row in content.iterrows():
                    if y < 50:
                        c.showPage()
                        y = height - 50
                    c.drawString(120, y, f"{idx}: {', '.join([f'{col}: {val:.2f}' for col, val in row.items()])}")
                    y -= 15
            else:  # Si c'est un graphique Plotly
                # Créer un fichier temporaire pour l'image
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
                    pio.write_image(content, tmpfile.name, format="png", width=500, height=250)
                    c.drawImage(tmpfile.name, 100, y - 250, width=400, height=200)
                # Supprimer le fichier temporaire après usage
                os.unlink(tmpfile.name)
                y -= 260

    c.save()

# Questions pour le quiz (exemple réduit)
easy_questions = [
    {"question": "Que sont les intérêts composés ?", "options": ["Intérêts sur le capital initial", "Intérêts sur le capital + intérêts cumulés", "Intérêts fixes"], "correct": "Intérêts sur le capital + intérêts cumulés", "explanation": "Les intérêts composés génèrent des intérêts sur le capital initial et sur les intérêts déjà accumulés."},
    {"question": "Qu’est-ce qu’un dividende ?", "options": ["Un prêt bancaire", "Une part des bénéfices distribuée", "Un taux d’intérêt"], "correct": "Une part des bénéfices distribuée", "explanation": "Un dividende est une partie des profits qu’une entreprise partage avec ses actionnaires."},
]

medium_questions = [
    {"question": "Que mesure la volatilité ?", "options": ["Le rendement moyen", "L’écart des rendements", "Le prix d’une action"], "correct": "L’écart des rendements", "explanation": "La volatilité indique à quel point les rendements d’un actif fluctuent autour de leur moyenne."},
    {"question": "Qu’est-ce que la VAN ?", "options": ["La valeur future d’un investissement", "La valeur actuelle des flux de trésorerie", "Le taux de rendement"], "correct": "La valeur actuelle des flux de trésorerie", "explanation": "La VAN actualise les flux futurs pour estimer leur valeur aujourd’hui."},
]

hard_questions = [
    {"question": "Que représente le TRI ?", "options": ["Le taux d’actualisation rendant la VAN nulle", "Le rendement moyen d’un portefeuille", "Le taux d’intérêt d’un prêt"], "correct": "Le taux d’actualisation rendant la VAN nulle", "explanation": "Le TRI est le taux qui équilibre les entrées et sorties de trésorerie dans un projet."},
    {"question": "Quel indicateur mesure le risque ajusté au rendement ?", "options": ["Ratio Sharpe", "Beta", "Volatilité"], "correct": "Ratio Sharpe", "explanation": "Le ratio Sharpe évalue le rendement excédentaire par unité de risque."},
]

# Liste des sections disponibles
SECTIONS = [
    "Accueil (KPI)",
    "Calculatrices Financières",
    "Analyse de Portefeuille",
    "Visualisation Boursière",
    "Prédiction de Prix",
    "Simulation Monte Carlo",
    "Analyse de Sentiments",
    "Quiz Financier",
    "Rapport Personnalisé"
]

# Initialisation de session_state pour l'authentification
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.user_info = {}
    st.session_state.preferred_section = None
    # Initialisation des attributs pour Analyse de Portefeuille
    st.session_state.portfolio_actifs = "AAPL, MSFT"
    st.session_state.portfolio_montants = "1000, 2000"
    st.session_state.portfolio_period = "1y"

# Page de connexion
if not st.session_state.authenticated:
    st.title("Bienvenue sur le Tableau de Bord Financier")
    st.subheader("Veuillez vous authentifier")
    with st.form(key='login_form'):
        nom = st.text_input("Nom")
        prenom = st.text_input("Prénom")
        age = st.number_input("Âge", min_value=1, max_value=120, step=1)
        email = st.text_input("Email (optionnel)", "")
        submit_button = st.form_submit_button(label="Se connecter")
        if submit_button and nom and prenom and age:
            st.session_state.user_info = {"nom": nom, "prenom": prenom, "age": age, "email": email if email else "Non fourni"}
            st.session_state.authenticated = True
            st.success(f"Connexion réussie, {prenom} !")
            st.rerun()
        else:
            st.error("Veuillez remplir tous les champs obligatoires (Nom, Prénom, Âge).")

elif st.session_state.authenticated and st.session_state.preferred_section is None:
    st.title(f"Bonjour, {st.session_state.user_info['prenom']} !")
    st.subheader("Que souhaitez-vous explorer aujourd’hui ?")
    preferred_section = st.selectbox("Choisissez une section", SECTIONS)
    if st.button("Confirmer"):
        st.session_state.preferred_section = preferred_section
        st.success(f"Vous avez choisi : {preferred_section}")
        st.rerun()

else:
    prenom = st.session_state.user_info['prenom']
    age = st.session_state.user_info['age']
    st.sidebar.title(f"Bienvenue, {prenom} !")
    st.sidebar.write(f"Âge : {age}")
    section = st.sidebar.radio("Choisir une section", SECTIONS, index=SECTIONS.index(st.session_state.preferred_section))
    
    if age < 30:
        st.sidebar.write("Astuce : Pensez à investir tôt pour profiter des intérêts composés !")
    elif age >= 50:
        st.sidebar.write("Astuce : Diversifiez pour réduire les risques à l’approche de la retraite.")

    # Section "Accueil (KPI)"
    if section == "Accueil (KPI)":
        st.title(f"Tableau de Bord Financier - {prenom}")
        st.subheader("Indicateurs Clés de Performance (KPI)")

        if "kpi_tickers" not in st.session_state:
            st.session_state.kpi_tickers = "AAPL, MSFT"
        if "kpi_values" not in st.session_state:
            st.session_state.kpi_values = "1000, 2000"
        if "kpi_period" not in st.session_state:
            st.session_state.kpi_period = "1y"

        col_input1, col_input2 = st.columns(2)
        with col_input1:
            portfolio_tickers = st.text_input("Actifs du portefeuille (ex: AAPL, MSFT)", value=st.session_state.kpi_tickers, key="kpi_tickers_input")
            st.session_state.kpi_tickers = portfolio_tickers
        with col_input2:
            portfolio_values = st.text_input("Montants investis (ex: 1000, 2000)", value=st.session_state.kpi_values, key="kpi_values_input")
            st.session_state.kpi_values = portfolio_values
        period = st.selectbox("Période des données", ["1mo", "3mo", "6mo", "1y", "2y"], index=["1mo", "3mo", "6mo", "1y", "2y"].index(st.session_state.kpi_period), key="kpi_period_input")
        st.session_state.kpi_period = period

        portfolio_tickers = portfolio_tickers.split(",")
        portfolio_values = portfolio_values.split(",")

        if len(portfolio_tickers) != len(portfolio_values):
            st.error("Le nombre d'actifs et de montants doit correspondre !")
        else:
            portfolio_values_raw = [v.strip() for v in portfolio_values]
            portfolio_values_num = []
            all_valid = True
            for val in portfolio_values_raw:
                if val.replace(".", "").isdigit() or (val.startswith("-") and val[1:].replace(".", "").isdigit()):
                    portfolio_values_num.append(float(val))
                else:
                    all_valid = False
                    st.error(f"'{val}' n'est pas un nombre valide. Utilisez des nombres (ex: 1000, 2000).")
                    break

            if all_valid:
                total_value = sum(portfolio_values_num)
                with st.spinner("Calcul des indicateurs..."):
                    returns, volatilities = [], []
                    for ticker in portfolio_tickers:
                        mean_ret, vol = get_stock_params(ticker.strip(), period)
                        returns.append(mean_ret)
                        volatilities.append(vol)
                    avg_return = np.mean(returns) * 100
                    avg_volatility = np.mean(volatilities) * 100

                col1, col2, col3 = st.columns(3)
                col1.metric("Valeur Totale", f"{total_value:.2f} €")
                col2.metric("Rendement Moyen Annualisé", f"{avg_return:.2f} %", delta=f"{avg_return:.2f}%", delta_color="normal")
                col3.metric("Volatilité Moyenne", f"{avg_volatility:.2f} %", delta_color="off")

                st.subheader("Aperçu des Prix")
                fig = px.line(title=f"Prix sur {period}")
                for ticker in portfolio_tickers:
                    df = get_stock_data(ticker.strip(), period)
                    if not df.empty:
                        fig.add_scatter(x=df.index, y=df["Close"], mode="lines", name=ticker)
                st.plotly_chart(fig, use_container_width=True)

                st.subheader("Actualités Financières")
                api_key = "b359db1354504094892656584102768b"
                newsapi = NewsApiClient(api_key=api_key)

                for ticker in portfolio_tickers:
                    with st.expander(f"Actualités pour {ticker.strip()} (jusqu'à 60 articles)", expanded=False):
                        try:
                            news = newsapi.get_everything(q=ticker.strip(), language='fr', sort_by='publishedAt', page_size=60)
                            articles = news['articles']
                            if articles:
                                df_news = pd.DataFrame({
                                    "Titre": [article['title'] for article in articles],
                                    "Date": [article['publishedAt'][:10] for article in articles],
                                    "Source": [article['source']['name'] for article in articles],
                                    "Lien": [f"[Lire]({article['url']})" for article in articles]
                                })
                                st.dataframe(df_news, use_container_width=True)
                            else:
                                st.write("Aucune actualité trouvée.")
                        except Exception as e:
                            st.error(f"Erreur pour {ticker} : {e}")

                st.subheader("Recherche d'Actualités Personnalisée")
                col_search1, col_search2, col_search3 = st.columns(3)
                with col_search1:
                    if "kpi_search_query" not in st.session_state:
                        st.session_state.kpi_search_query = ""
                    search_query = st.text_input("Ticker ou mot-clé (ex: TSLA, Bitcoin)", value=st.session_state.kpi_search_query, key="kpi_search_query_input")
                    st.session_state.kpi_search_query = search_query
                with col_search2:
                    if "kpi_date_from" not in st.session_state:
                        st.session_state.kpi_date_from = datetime.now() - timedelta(days=30)
                    date_from = st.date_input("À partir de", value=st.session_state.kpi_date_from, key="kpi_date_from_input")
                    st.session_state.kpi_date_from = date_from
                with col_search3:
                    if "kpi_sources" not in st.session_state:
                        st.session_state.kpi_sources = []
                    sources = st.multiselect("Sources (optionnel)", ["Le Monde", "Les Echos", "Reuters", "AFP"], default=st.session_state.kpi_sources, key="kpi_sources_input")
                    st.session_state.kpi_sources = sources
                
                if st.button("Rechercher") and search_query:
                    with st.spinner("Recherche en cours..."):
                        try:
                            source_str = ",".join([s.lower().replace(" ", "-") for s in sources]) if sources else None
                            news = newsapi.get_everything(
                                q=search_query.strip(),
                                language='fr',
                                sort_by='publishedAt',
                                page_size=60,
                                from_param=date_from.strftime("%Y-%m-%d"),
                                sources=source_str
                            )
                            articles = news['articles']
                            if articles:
                                df_search = pd.DataFrame({
                                    "Titre": [article['title'] for article in articles],
                                    "Date": [article['publishedAt'][:10] for article in articles],
                                    "Source": [article['source']['name'] for article in articles],
                                    "Lien": [f"[Lire]({article['url']})" for article in articles]
                                })
                                st.write(f"Résultats pour '{search_query}' (jusqu'à 60 articles) :")
                                st.dataframe(df_search, use_container_width=True)
                            else:
                                st.write(f"Aucune actualité trouvée pour '{search_query}'.")
                        except Exception as e:
                            st.error(f"Erreur lors de la recherche : {e}")

    # Section "Calculatrices Financières"
    elif section == "Calculatrices Financières":
        st.title(f"Calculatrices Financières - {prenom}")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("Intérêts Composés")
            if "calc_principal" not in st.session_state:
                st.session_state.calc_principal = 1000.0
            if "calc_taux" not in st.session_state:
                st.session_state.calc_taux = 5.0
            if "calc_annees" not in st.session_state:
                st.session_state.calc_annees = 10
            principal = st.number_input("Montant initial (€)", min_value=0.0, value=st.session_state.calc_principal, key="calc_principal_input")
            taux = st.slider("Taux annuel (%)", 0.0, 20.0, st.session_state.calc_taux, key="calc_taux_input")
            annees = st.slider("Durée (années)", 1, 30, st.session_state.calc_annees, key="calc_annees_input")
            st.session_state.calc_principal = principal
            st.session_state.calc_taux = taux
            st.session_state.calc_annees = annees
            resultat = calcul_interets_composes(principal, taux, annees)
            st.write(f"Valeur future : **{resultat:.2f} €**")
        with col2:
            st.subheader("Amortissement de Prêt")
            if "pret_principal" not in st.session_state:
                st.session_state.pret_principal = 50000.0
            if "pret_taux" not in st.session_state:
                st.session_state.pret_taux = 3.0
            if "pret_annees" not in st.session_state:
                st.session_state.pret_annees = 15
            pret_principal = st.number_input("Montant du prêt (€)", min_value=0.0, value=st.session_state.pret_principal, key="pret_principal_input")
            pret_taux = st.slider("Taux annuel (%) ", 0.0, 15.0, st.session_state.pret_taux, key="pret_taux_input")
            pret_annees = st.slider("Durée (années) ", 1, 30, st.session_state.pret_annees, key="pret_annees_input")
            st.session_state.pret_principal = pret_principal
            st.session_state.pret_taux = pret_taux
            st.session_state.pret_annees = pret_annees
            paiement = calcul_amortissement_pret(pret_principal, pret_taux, pret_annees)
            st.write(f"Paiement mensuel : **{paiement:.2f} €**")
        with col3:
            st.subheader("VAN et TRI")
            if "flux_input" not in st.session_state:
                st.session_state.flux_input = "-5000, 2000, 3000"
            if "van_taux" not in st.session_state:
                st.session_state.van_taux = 5.0
            flux_input = st.text_area("Flux (ex: -5000, 2000, 3000)", value=st.session_state.flux_input, key="flux_input_input")
            st.session_state.flux_input = flux_input
            flux_raw = [x.strip() for x in flux_input.split(",")]
            flux = []
            all_valid = True
            for f in flux_raw:
                if f.replace(".", "").isdigit() or (f.startswith("-") and f[1:].replace(".", "").isdigit()):
                    flux.append(float(f))
                else:
                    all_valid = False
                    st.error(f"'{f}' n'est pas un nombre valide.")
                    break
            if all_valid:
                van_taux = st.slider("Taux d'actualisation (%)", 0.0, 20.0, st.session_state.van_taux, key="van_taux_input")
                st.session_state.van_taux = van_taux
                van = calcul_van(flux, van_taux)
                tri = calcul_tri(flux)
                st.write(f"VAN : **{van:.2f} €**")
                st.write(f"TRI : **{tri:.2f} %**")

    # Section "Analyse de Portefeuille"
    elif section == "Analyse de Portefeuille":
        st.title(f"Analyse de Portefeuille - {prenom}")
        
        st.subheader("Composition du portefeuille")
        col1, col2 = st.columns(2)
        with col1:
            actifs = st.text_input("Actifs (ex: AAPL, MSFT)", value=st.session_state.portfolio_actifs, key="portfolio_actifs_input")
            st.session_state.portfolio_actifs = actifs
        with col2:
            montants = st.text_input("Montants investis (ex: 1000, 2000)", value=st.session_state.portfolio_montants, key="portfolio_montants_input")
            st.session_state.portfolio_montants = montants
        period = st.selectbox("Période d'analyse", ["1mo", "3mo", "6mo", "1y", "2y"], index=["1mo", "3mo", "6mo", "1y", "2y"].index(st.session_state.portfolio_period), key="portfolio_period_input")
        st.session_state.portfolio_period = period

        actifs_list = [x.strip().upper() for x in actifs.split(",")]
        montants_list_raw = [x.strip() for x in montants.split(",")]

        montants_list = []
        all_valid = True
        for montant in montants_list_raw:
            if montant.replace(".", "").isdigit() or (montant.startswith("-") and montant[1:].replace(".", "").isdigit()):
                montants_list.append(float(montant))
            else:
                all_valid = False
                st.error(f"'{montant}' n'est pas un nombre valide. Utilisez des nombres (ex: 1000, 2000).")
                break

        if all_valid and len(actifs_list) == len(montants_list):
            df_portfolio = pd.DataFrame({"Actif": actifs_list, "Montant Initial": montants_list})

            st.subheader("Valeur Actuelle et Performance")
            current_values = []
            historical_data = {}
            for ticker in actifs_list:
                df = get_stock_data(ticker, period)
                if not df.empty:
                    current_price = df['Close'][-1]
                    initial_amount = df_portfolio[df_portfolio["Actif"] == ticker]["Montant Initial"].values[0]
                    current_value = (initial_amount / df['Close'][0]) * current_price
                    current_values.append(current_value)
                    historical_data[ticker] = df['Close']
                else:
                    current_values.append(df_portfolio[df_portfolio["Actif"] == ticker]["Montant Initial"].values[0])
                    st.warning(f"Données indisponibles pour {ticker}. Valeur initiale utilisée.")
            df_portfolio["Valeur Actuelle"] = current_values
            total_initial = df_portfolio["Montant Initial"].sum()
            total_current = df_portfolio["Valeur Actuelle"].sum()
            rendement_total = ((total_current - total_initial) / total_initial) * 100
            rendement_annualise = ((total_current / total_initial) ** (1 / (int(period[:-2]) / 12)) - 1) * 100 if period.endswith("mo") else ((total_current / total_initial) ** (1 / int(period[:-1])) - 1) * 100

            col1, col2, col3 = st.columns(3)
            col1.metric("Valeur Initiale", f"{total_initial:.2f} €")
            col2.metric("Valeur Actuelle", f"{total_current:.2f} €")
            col3.metric("Rendement Total", f"{rendement_total:.2f} %", delta=f"{rendement_annualise:.2f}% annualisé")

            st.dataframe(df_portfolio, use_container_width=True)

            st.subheader("Répartition par Secteur")
            sectors = {}
            for ticker in actifs_list:
                stock = yf.Ticker(ticker)
                info = stock.info
                sector = info.get("sector", "Inconnu")
                sectors[ticker] = sector
            df_portfolio["Secteur"] = [sectors[ticker] for ticker in df_portfolio["Actif"]]
            sector_dist = df_portfolio.groupby("Secteur")["Valeur Actuelle"].sum().reset_index()
            fig_sector = px.pie(sector_dist, values="Valeur Actuelle", names="Secteur", title="Répartition par Secteur")
            st.plotly_chart(fig_sector, use_container_width=True)

            st.subheader("Évolution Temporelle")
            df_historical = pd.DataFrame(historical_data)
            df_historical.index = df_historical.index.tz_localize(None)
            portfolio_value = pd.DataFrame(index=df_historical.index)
            for ticker in actifs_list:
                initial_amount = df_portfolio[df_portfolio["Actif"] == ticker]["Montant Initial"].values[0]
                shares = initial_amount / df_historical[ticker][0]
                portfolio_value[ticker] = df_historical[ticker] * shares
            portfolio_value["Total"] = portfolio_value.sum(axis=1)
            fig_time = px.line(portfolio_value, x=portfolio_value.index, y="Total", title=f"Évolution du Portefeuille ({period})")
            st.plotly_chart(fig_time, use_container_width=True)

            st.subheader("Risque du Portefeuille")
            returns = df_historical.pct_change().dropna()
            portfolio_returns = (returns * (df_portfolio["Montant Initial"] / total_initial).values).sum(axis=1)
            volatility = portfolio_returns.std() * np.sqrt(252) * 100
            st.metric("Volatilité Annualisée", f"{volatility:.2f} %")

            st.subheader("Corrélation entre Actifs")
            correlation_matrix = returns.corr()
            fig_corr = px.imshow(correlation_matrix, text_auto=True, title="Matrice de Corrélation", color_continuous_scale="RdBu")
            st.plotly_chart(fig_corr, use_container_width=True)

            st.subheader("Répartition Initiale")
            fig_pie = px.pie(df_portfolio, values="Montant Initial", names="Actif", title="Répartition Initiale")
            st.plotly_chart(fig_pie, use_container_width=True)

            # Nouvelle section : Frontière Efficiente et Portefeuille Optimal
            st.subheader("Frontière Efficiente et Portefeuille Optimal")
            
            # Calcul des rendements moyens annualisés et matrice de covariance
            daily_returns = returns.dropna()
            mean_returns = daily_returns.mean() * 252  # Rendements annualisés
            cov_matrix = daily_returns.cov() * 252  # Matrice de covariance annualisée

            num_assets = len(actifs_list)
            num_portfolios = 10000  # Nombre de portefeuilles simulés

            # Simulation de portefeuilles aléatoires
            np.random.seed(42)
            portfolio_returns = []
            portfolio_volatilities = []
            portfolio_weights = []

            for _ in range(num_portfolios):
                weights = np.random.random(num_assets)
                weights /= np.sum(weights)  # Normalisation pour que la somme = 1
                portfolio_weights.append(weights)
                port_return = np.sum(weights * mean_returns)
                port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                portfolio_returns.append(port_return * 100)  # En pourcentage
                portfolio_volatilities.append(port_volatility * 100)  # En pourcentage

            # Création d'un DataFrame pour les portefeuilles simulés
            portfolios = pd.DataFrame({
                "Rendement (%)": portfolio_returns,
                "Volatilité (%)": portfolio_volatilities
            })

            # Identification du portefeuille à variance minimale
            min_vol_idx = portfolios["Volatilité (%)"].idxmin()
            min_vol_weights = portfolio_weights[min_vol_idx]
            min_vol_return = portfolios["Rendement (%)"][min_vol_idx]
            min_vol_volatility = portfolios["Volatilité (%)"][min_vol_idx]

            # Calcul de la frontière efficiente
            target_returns = np.linspace(min(portfolio_returns), max(portfolio_returns), 50)
            efficient_volatilities = []

            def portfolio_volatility(weights, cov_matrix):
                return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

            def minimize_volatility(target_return, mean_returns, cov_matrix):
                constraints = (
                    {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Somme des poids = 1
                    {'type': 'eq', 'fun': lambda x: np.sum(x * mean_returns) - target_return}  # Rendement cible
                )
                bounds = tuple((0, 1) for _ in range(num_assets))  # Poids entre 0 et 1
                initial_weights = np.array([1/num_assets] * num_assets)
                result = minimize(portfolio_volatility, initial_weights, args=(cov_matrix,),
                                method='SLSQP', bounds=bounds, constraints=constraints)
                return result.fun

            for target in target_returns:
                efficient_vol = minimize_volatility(target / 100, mean_returns, cov_matrix) * 100  # Convertir en pourcentage
                efficient_volatilities.append(efficient_vol)

            # Visualisation de la frontière efficiente
            fig_eff = px.scatter(portfolios, x="Volatilité (%)", y="Rendement (%)", 
                                title="Frontière Efficiente et Portefeuilles Simulés",
                                labels={"Volatilité (%)": "Volatilité Annualisée (%)", "Rendement (%)": "Rendement Attendu (%)"},
                                opacity=0.3)
            fig_eff.add_scatter(x=efficient_volatilities, y=target_returns, mode='lines', name='Frontière Efficiente', line=dict(color='green'))
            fig_eff.add_scatter(x=[min_vol_volatility], y=[min_vol_return], mode='markers', name='Portefeuille Variance Minimale', 
                                marker=dict(color='red', size=10))
            st.plotly_chart(fig_eff, use_container_width=True)

            # Affichage des détails du portefeuille à variance minimale
            st.write("**Portefeuille à Variance Minimale :**")
            min_vol_df = pd.DataFrame({
                "Actif": actifs_list,
                "Poids (%)": [w * 100 for w in min_vol_weights]
            })
            st.dataframe(min_vol_df, use_container_width=True)
            st.write(f"Rendement Attendu : **{min_vol_return:.2f}%**")
            st.write(f"Volatilité : **{min_vol_volatility:.2f}%**")

        elif all_valid:
            st.error("Le nombre d'actifs et de montants doit correspondre !")

    # Section "Visualisation Boursière"
    elif section == "Visualisation Boursière":
        st.title(f"Visualisation Boursière - {prenom}")
        if "visu_tickers" not in st.session_state:
            st.session_state.visu_tickers = "AAPL, MSFT"
        if "visu_period" not in st.session_state:
            st.session_state.visu_period = "1y"
        tickers = st.text_input("Symboles (ex: AAPL, MSFT)", value=st.session_state.visu_tickers, key="visu_tickers_input")
        st.session_state.visu_tickers = tickers
        period = st.selectbox("Période", ["1mo", "3mo", "6mo", "1y", "2y"], index=["1mo", "3mo", "6mo", "1y", "2y"].index(st.session_state.visu_period), key="visu_period_input")
        st.session_state.visu_period = period
        tickers = tickers.split(",")
        with st.spinner("Chargement..."):
            fig = px.line(title="Prix de clôture")
            for ticker in tickers:
                df = get_stock_data(ticker.strip(), period)
                if not df.empty:
                    fig.add_scatter(x=df.index, y=df["Close"], mode="lines", name=ticker)
            st.plotly_chart(fig)

    # Section "Prédiction de Prix"
    elif section == "Prédiction de Prix":
        st.title(f"Prédiction de Prix - {prenom}")
        if "pred_ticker" not in st.session_state:
            st.session_state.pred_ticker = "AAPL"
        if "pred_model" not in st.session_state:
            st.session_state.pred_model = "Régression Linéaire"
        ticker_pred = st.text_input("Symbole (ex: AAPL)", value=st.session_state.pred_ticker, key="pred_ticker_input")
        st.session_state.pred_ticker = ticker_pred
        model_choice = st.selectbox("Modèle", ["Régression Linéaire", "Prophet"], index=["Régression Linéaire", "Prophet"].index(st.session_state.pred_model), key="pred_model_input")
        st.session_state.pred_model = model_choice
        df_pred = get_stock_data(ticker_pred)
        if not df_pred.empty:
            with st.spinner("Calcul..."):
                if model_choice == "Régression Linéaire":
                    df_pred['Days'] = np.arange(len(df_pred))
                    X = df_pred[['Days']]
                    y = df_pred['Close']
                    model = LinearRegression()
                    model.fit(X, y)
                    future_days = np.arange(len(df_pred), len(df_pred) + 30).reshape(-1, 1)
                    predictions = model.predict(future_days)
                    dates_future = pd.date_range(start=df_pred.index[-1], periods=31, freq='B')[1:]
                    explanation = "Tendance linéaire simple."
                else:
                    df_prophet = df_pred.reset_index()[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
                    df_prophet['ds'] = df_prophet['ds'].dt.tz_localize(None)
                    model = Prophet(daily_seasonality=True)
                    model.fit(df_prophet)
                    future = model.make_future_dataframe(periods=30, freq='B')
                    forecast = model.predict(future)
                    predictions = forecast['yhat'].tail(30)
                    dates_future = forecast['ds'].tail(30)
                    explanation = "Tendances et saisonnalités."
                fig_pred = px.line(title=f"Prédiction {ticker_pred} ({model_choice})")
                fig_pred.add_scatter(x=df_pred.index, y=df_pred['Close'], mode='lines', name='Historique')
                fig_pred.add_scatter(x=dates_future, y=predictions, mode='lines', name='Prédiction', line=dict(dash='dash'))
                st.plotly_chart(fig_pred)
                st.write(f"**Explication** : {explanation}")

    # Section "Simulation Monte Carlo"
    elif section == "Simulation Monte Carlo":
        st.title(f"Simulation Monte Carlo - {prenom}")
        if "mc_invest" not in st.session_state:
            st.session_state.mc_invest = 10000.0
        if "mc_years" not in st.session_state:
            st.session_state.mc_years = 5
        if "mc_traj" not in st.session_state:
            st.session_state.mc_traj = 10
        if "mc_mode" not in st.session_state:
            st.session_state.mc_mode = "Basé sur un titre"
        if "mc_ticker" not in st.session_state:
            st.session_state.mc_ticker = "AAPL"
        if "mc_period" not in st.session_state:
            st.session_state.mc_period = "1y"
        if "mc_mean_ret" not in st.session_state:
            st.session_state.mc_mean_ret = 5.0
        if "mc_vol" not in st.session_state:
            st.session_state.mc_vol = 15.0

        invest = st.number_input("Investissement (€)", min_value=0.0, value=st.session_state.mc_invest, key="mc_invest_input")
        st.session_state.mc_invest = invest
        sim_years = st.slider("Durée (années)", 1, 20, st.session_state.mc_years, key="mc_years_input")
        st.session_state.mc_years = sim_years
        num_traj = st.slider("Trajectoires", 1, 50, st.session_state.mc_traj, key="mc_traj_input")
        st.session_state.mc_traj = num_traj
        mode = st.radio("Mode", ["Basé sur un titre", "Manuel"], index=["Basé sur un titre", "Manuel"].index(st.session_state.mc_mode), key="mc_mode_input")
        st.session_state.mc_mode = mode
        if mode == "Basé sur un titre":
            ticker = st.text_input("Symbole (ex: AAPL)", value=st.session_state.mc_ticker, key="mc_ticker_input")
            st.session_state.mc_ticker = ticker
            period = st.selectbox("Période", ["1y", "2y", "5y"], index=["1y", "2y", "5y"].index(st.session_state.mc_period), key="mc_period_input")
            st.session_state.mc_period = period
            mean_ret, vol = get_stock_params(ticker, period)
        else:
            mean_ret = st.slider("Rendement (%)", 0.0, 20.0, st.session_state.mc_mean_ret, key="mc_mean_ret_input") / 100
            vol = st.slider("Volatilité (%)", 0.0, 50.0, st.session_state.mc_vol, key="mc_vol_input") / 100
            st.session_state.mc_mean_ret = mean_ret * 100
            st.session_state.mc_vol = vol * 100
        paths = monte_carlo_simulation(invest, mean_ret, vol, sim_years)
        df_paths = pd.DataFrame(paths, columns=[f"Sim {i+1}" for i in range(paths.shape[1])])
        df_paths['Days'] = np.arange(252 * sim_years)
        df_paths['Mean'] = df_paths.iloc[:, :-1].mean(axis=1)
        cols_to_plot = df_paths.columns[:num_traj].tolist() + ['Mean']
        fig_mc = px.line(df_paths, x="Days", y=cols_to_plot, title="Simulation Monte Carlo")
        st.plotly_chart(fig_mc)

        # Ajout de l'explication des résultats
        final_values = paths[-1]  # Valeurs finales des simulations
        worst_case = np.min(final_values)
        average_case = np.mean(final_values)
        best_case = np.max(final_values)
        st.subheader("Explication des Résultats")
        st.write(f"Voici les résultats de la simulation Monte Carlo sur {sim_years} ans avec un investissement initial de {invest:.2f} € :")
        st.write(f"- **Pire scénario** : {worst_case:.2f} € (le plus bas des résultats simulés).")
        st.write(f"- **Scénario moyen** : {average_case:.2f} € (moyenne des trajectoires).")
        st.write(f"- **Meilleur scénario** : {best_case:.2f} € (le plus haut des résultats simulés).")
        st.write("Ces valeurs illustrent l'incertitude liée à l'évolution future en fonction du rendement et de la volatilité choisis.")

    # Section "Analyse de Sentiments"
    elif section == "Analyse de Sentiments":
        st.title(f"Analyse de Sentiments - {prenom}")
        if "sentiment_text" not in st.session_state:
            st.session_state.sentiment_text = "Le marché est en hausse !"
        texte = st.text_area("Texte", value=st.session_state.sentiment_text, key="sentiment_text_input")
        st.session_state.sentiment_text = texte
        sentiment = analyze_sentiment(texte)
        st.write(f"Sentiment : **{'Positif' if sentiment > 0 else 'Négatif' if sentiment < 0 else 'Neutre'}** (Score: {sentiment:.2f})")

    # Section "Quiz Financier"
    elif section == "Quiz Financier":
        st.title(f"Quiz Financier - {prenom}")
        if "quiz_difficulty" not in st.session_state:
            st.session_state.quiz_difficulty = "Facile"
        difficulty = st.selectbox("Niveau", ["Facile", "Moyen", "Difficile"], index=["Facile", "Moyen", "Difficile"].index(st.session_state.quiz_difficulty), key="quiz_difficulty_input")
        st.session_state.quiz_difficulty = difficulty
        question_sets = {"Facile": easy_questions, "Moyen": medium_questions, "Difficile": hard_questions}
        selected_questions = question_sets[difficulty]

        # Initialisation du quiz
        if 'quiz_questions' not in st.session_state or st.session_state.quiz_difficulty != difficulty:
            num_questions = min(10, len(selected_questions))
            if num_questions == 0:
                st.error("Aucune question disponible pour ce niveau !")
            else:
                st.session_state.quiz_questions = random.sample(selected_questions, num_questions)
                st.session_state.current_question = 0
                st.session_state.score = 0
                st.session_state.quiz_finished = False
                st.session_state.timer_start = None

        if not st.session_state.quiz_finished and 'quiz_questions' in st.session_state:
            q_index = st.session_state.current_question
            total_questions = len(st.session_state.quiz_questions)

            if q_index >= total_questions:
                st.session_state.quiz_finished = True
                st.rerun()

            question_data = st.session_state.quiz_questions[q_index]
            st.subheader(f"Question {q_index + 1}/{total_questions}")
            st.write(question_data["question"])
            if st.session_state.timer_start is None:
                st.session_state.timer_start = time.time()
            time_left = max(0, 30 - (time.time() - st.session_state.timer_start))
            st.write(f"Temps restant : {int(time_left)}s")
            user_answer = st.radio("Réponse", question_data["options"], key=f"q{q_index}")
            if st.button("Soumettre", key=f"submit{q_index}") or time_left <= 0:
                if time_left <= 0:
                    st.error("Temps écoulé !")
                elif user_answer == question_data["correct"]:
                    st.success("Correct !")
                    st.session_state.score += 1
                else:
                    st.error(f"Faux. Réponse : {question_data['correct']}")
                st.write(f"Explication : {question_data['explanation']}")
                st.session_state.current_question += 1
                st.session_state.timer_start = None
                if st.session_state.current_question >= total_questions:
                    st.session_state.quiz_finished = True
                st.rerun()
        if st.session_state.quiz_finished:
            st.subheader("Quiz Terminé !")
            score = st.session_state.score
            total_questions = len(st.session_state.quiz_questions)
            st.write(f"Votre score : **{score}/{total_questions}** ({(score / total_questions) * 100:.1f}%)")
            if st.button("Recommencer"):
                del st.session_state.quiz_questions
                del st.session_state.current_question
                del st.session_state.score
                del st.session_state.quiz_finished
                del st.session_state.timer_start
                st.rerun()

    # Section "Rapport Personnalisé"
    elif section == "Rapport Personnalisé":
        st.title(f"Générateur de Rapport Personnalisé - {prenom}")
        st.subheader("Que voulez-vous inclure dans votre rapport ?")
        
        inclure_interets = st.checkbox("Intérêts Composés")
        inclure_pret = st.checkbox("Amortissement de Prêt")
        inclure_van = st.checkbox("Valeur Actuelle Nette (VAN)")
        inclure_tri = st.checkbox("Taux de Rentabilité Interne (TRI)")
        inclure_kpi = st.checkbox("Indicateurs Clés (KPI)")
        inclure_sentiment = st.checkbox("Analyse de Sentiment")
        inclure_graphique = st.checkbox("Graphique Boursier")
        inclure_monte_carlo = st.checkbox("Résultats Monte Carlo")
        inclure_portefeuille = st.checkbox("Analyse de Portefeuille (avec graphiques)")

        selected_data = {}
        graphs = {}

        user_info = st.session_state.user_info

        if inclure_interets:
            if "report_principal" not in st.session_state:
                st.session_state.report_principal = 1000.0
            if "report_taux" not in st.session_state:
                st.session_state.report_taux = 5.0
            if "report_annees" not in st.session_state:
                st.session_state.report_annees = 10
            principal = st.number_input("Montant initial (€)", min_value=0.0, value=st.session_state.report_principal, key="report_principal_input")
            taux = st.slider("Taux annuel (%)", 0.0, 20.0, st.session_state.report_taux, key="report_taux_input")
            annees = st.slider("Durée (années)", 1, 30, st.session_state.report_annees, key="report_annees_input")
            st.session_state.report_principal = principal
            st.session_state.report_taux = taux
            st.session_state.report_annees = annees
            resultat = calcul_interets_composes(principal, taux, annees)
            selected_data["Intérêts Composés"] = generate_enriched_text("Valeur future des intérêts", resultat, "Montant final")

        if inclure_pret:
            if "report_pret_principal" not in st.session_state:
                st.session_state.report_pret_principal = 50000.0
            if "report_pret_taux" not in st.session_state:
                st.session_state.report_pret_taux = 3.0
            if "report_pret_annees" not in st.session_state:
                st.session_state.report_pret_annees = 15
            pret_principal = st.number_input("Montant du prêt (€)", min_value=0.0, value=st.session_state.report_pret_principal, key="report_pret_principal_input")
            pret_taux = st.slider("Taux annuel (%)", 0.0, 15.0, st.session_state.report_pret_taux, key="report_pret_taux_input")
            pret_annees = st.slider("Durée (années)", 1, 30, st.session_state.report_pret_annees, key="report_pret_annees_input")
            st.session_state.report_pret_principal = pret_principal
            st.session_state.report_pret_taux = pret_taux
            st.session_state.report_pret_annees = pret_annees
            paiement = calcul_amortissement_pret(pret_principal, pret_taux, pret_annees)
            selected_data["Amortissement de Prêt"] = generate_enriched_text("Paiement mensuel du prêt", paiement, "Paiement mensuel")

        if inclure_van or inclure_tri:
            if "report_flux_input" not in st.session_state:
                st.session_state.report_flux_input = "-5000, 2000, 3000"
            flux_input = st.text_input("Flux de trésorerie (ex: -5000, 2000, 3000)", value=st.session_state.report_flux_input, key="report_flux_input_input")
            st.session_state.report_flux_input = flux_input
            flux_raw = [x.strip() for x in flux_input.split(",")]
            flux = []
            all_valid = True
            for f in flux_raw:
                if f.replace(".", "").isdigit() or (f.startswith("-") and f[1:].replace(".", "").isdigit()):
                    flux.append(float(f))
                else:
                    all_valid = False
                    st.error(f"'{f}' n'est pas un nombre valide.")
                    break
            if all_valid:
                if inclure_van:
                    if "report_van_taux" not in st.session_state:
                        st.session_state.report_van_taux = 5.0
                    van_taux = st.slider("Taux d'actualisation (%)", 0.0, 20.0, st.session_state.report_van_taux, key="report_van_taux_input")
                    st.session_state.report_van_taux = van_taux
                    van = calcul_van(flux, van_taux)
                    selected_data["VAN"] = generate_enriched_text("Valeur actuelle nette", van, "VAN")
                if inclure_tri:
                    tri = calcul_tri(flux)
                    selected_data["TRI"] = generate_enriched_text("Taux de rentabilité interne", tri, "TRI")

        if inclure_kpi:
            if "report_kpi_tickers" not in st.session_state:
                st.session_state.report_kpi_tickers = "AAPL, MSFT"
            if "report_kpi_values" not in st.session_state:
                st.session_state.report_kpi_values = "1000, 2000"
            portfolio_tickers = st.text_input("Actifs (ex: AAPL, MSFT)", value=st.session_state.report_kpi_tickers, key="report_kpi_tickers_input")
            portfolio_values = st.text_input("Montants (ex: 1000, 2000)", value=st.session_state.report_kpi_values, key="report_kpi_values_input")
            st.session_state.report_kpi_tickers = portfolio_tickers
            st.session_state.report_kpi_values = portfolio_values
            portfolio_tickers = portfolio_tickers.split(",")
            portfolio_values = portfolio_values.split(",")
            if len(portfolio_tickers) == len(portfolio_values):
                portfolio_values_raw = [v.strip() for v in portfolio_values]
                portfolio_values_num = []
                all_valid = True
                for val in portfolio_values_raw:
                    if val.replace(".", "").isdigit() or (val.startswith("-") and val[1:].replace(".", "").isdigit()):
                        portfolio_values_num.append(float(val))
                    else:
                        all_valid = False
                        st.error(f"'{val}' n'est pas un nombre valide.")
                        break
                if all_valid:
                    total_value = sum(portfolio_values_num)
                    returns, volatilities = [], []
                    for ticker in portfolio_tickers:
                        mean_ret, vol = get_stock_params(ticker.strip())
                        returns.append(mean_ret)
                        volatilities.append(vol)
                    avg_return = np.mean(returns) * 100
                    avg_volatility = np.mean(volatilities) * 100
                    selected_data["KPI"] = {
                        "Valeur Totale": f"{total_value:.2f} €",
                        "Rendement Moyen": f"{avg_return:.2f} %",
                        "Volatilité Moyenne": f"{avg_volatility:.2f} %"
                    }
            else:
                st.error("Le nombre d'actifs et de montants doit correspondre !")

        if inclure_sentiment:
            if "report_sentiment_text" not in st.session_state:
                st.session_state.report_sentiment_text = "Le marché est en hausse !"
            texte = st.text_area("Texte pour analyse", value=st.session_state.report_sentiment_text, key="report_sentiment_text_input")
            st.session_state.report_sentiment_text = texte
            sentiment = analyze_sentiment(texte)
            selected_data["Sentiment"] = f"Score : {sentiment:.2f} ({'Positif' if sentiment > 0 else 'Négatif' if sentiment < 0 else 'Neutre'})"

        if inclure_graphique:
            if "report_graph_ticker" not in st.session_state:
                st.session_state.report_graph_ticker = "AAPL"
            if "report_graph_period" not in st.session_state:
                st.session_state.report_graph_period = "1y"
            ticker = st.text_input("Symbole (ex: AAPL)", value=st.session_state.report_graph_ticker, key="report_graph_ticker_input")
            period = st.selectbox("Période", ["1mo", "3mo", "6mo", "1y", "2y"], index=["1mo", "3mo", "6mo", "1y", "2y"].index(st.session_state.report_graph_period), key="report_graph_period_input")
            st.session_state.report_graph_ticker = ticker
            st.session_state.report_graph_period = period
            df = get_stock_data(ticker, period)
            if not df.empty:
                fig = px.line(df, x=df.index, y="Close", title=f"Performance de {ticker}")
                st.plotly_chart(fig)
                selected_data["Graphique Boursier"] = f"Graphique de {ticker} pour {period} inclus."
                graphs["Performance Boursière"] = fig

        if inclure_monte_carlo:
            if "report_mc_invest" not in st.session_state:
                st.session_state.report_mc_invest = 10000.0
            if "report_mc_years" not in st.session_state:
                st.session_state.report_mc_years = 5
            if "report_mc_mean_ret" not in st.session_state:
                st.session_state.report_mc_mean_ret = 5.0
            if "report_mc_vol" not in st.session_state:
                st.session_state.report_mc_vol = 15.0
            invest = st.number_input("Investissement (€)", min_value=0.0, value=st.session_state.report_mc_invest, key="report_mc_invest_input")
            sim_years = st.slider("Durée (années)", 1, 20, st.session_state.report_mc_years, key="report_mc_years_input")
            mean_ret = st.slider("Rendement (%)", 0.0, 20.0, st.session_state.report_mc_mean_ret, key="report_mc_mean_ret_input") / 100
            vol = st.slider("Volatilité (%)", 0.0, 50.0, st.session_state.report_mc_vol, key="report_mc_vol_input") / 100
            st.session_state.report_mc_invest = invest
            st.session_state.report_mc_years = sim_years
            st.session_state.report_mc_mean_ret = mean_ret * 100
            st.session_state.report_mc_vol = vol * 100
            paths = monte_carlo_simulation(invest, mean_ret, vol, sim_years)
            final_values = paths[-1]
            worst_case = np.min(final_values)
            average_case = np.mean(final_values)
            best_case = np.max(final_values)
            selected_data["Monte Carlo"] = {
                "Moyenne": f"{average_case:.2f} €",
                "5e Percentile": f"{np.percentile(final_values, 5):.2f} €",
                "95e Percentile": f"{np.percentile(final_values, 95):.2f} €",
                "Explication": f"Pire scénario : {worst_case:.2f} €, Moyenne : {average_case:.2f} €, Meilleur scénario : {best_case:.2f} €"
            }
            df_paths = pd.DataFrame(paths, columns=[f"Sim {i+1}" for i in range(paths.shape[1])])
            df_paths['Days'] = np.arange(252 * sim_years)
            df_paths['Mean'] = df_paths.iloc[:, :-1].mean(axis=1)
            fig_mc = px.line(df_paths, x="Days", y="Mean", title="Simulation Monte Carlo (Moyenne)")
            st.plotly_chart(fig_mc)
            graphs["Simulation Monte Carlo"] = fig_mc

        if inclure_portefeuille:
            actifs = st.session_state.portfolio_actifs
            montants = st.session_state.portfolio_montants
            period = st.session_state.portfolio_period
            actifs_list = [x.strip().upper() for x in actifs.split(",")]
            montants_list_raw = [x.strip() for x in montants.split(",")]
            montants_list = []
            all_valid = True
            for montant in montants_list_raw:
                if montant.replace(".", "").isdigit() or (montant.startswith("-") and montant[1:].replace(".", "").isdigit()):
                    montants_list.append(float(montant))
                else:
                    all_valid = False
                    st.error(f"'{montant}' n'est pas un nombre valide.")
                    break
            if all_valid and len(actifs_list) == len(montants_list):
                df_portfolio = pd.DataFrame({"Actif": actifs_list, "Montant Initial": montants_list})
                current_values = []
                historical_data = {}
                for ticker in actifs_list:
                    df = get_stock_data(ticker, period)
                    if not df.empty:
                        current_price = df['Close'][-1]
                        initial_amount = df_portfolio[df_portfolio["Actif"] == ticker]["Montant Initial"].values[0]
                        current_value = (initial_amount / df['Close'][0]) * current_price
                        current_values.append(current_value)
                        historical_data[ticker] = df['Close']
                    else:
                        current_values.append(df_portfolio[df_portfolio["Actif"] == ticker]["Montant Initial"].values[0])
                df_portfolio["Valeur Actuelle"] = current_values
                total_initial = df_portfolio["Montant Initial"].sum()
                total_current = df_portfolio["Valeur Actuelle"].sum()
                rendement_total = ((total_current - total_initial) / total_initial) * 100
                rendement_annualise = ((total_current / total_initial) ** (1 / (int(period[:-2]) / 12)) - 1) * 100 if period.endswith("mo") else ((total_current / total_initial) ** (1 / int(period[:-1])) - 1) * 100

                # Données pour le rapport
                selected_data["Analyse de Portefeuille"] = {
                    "Valeur Initiale": f"{total_initial:.2f} €",
                    "Valeur Actuelle": f"{total_current:.2f} €",
                    "Rendement Total": f"{rendement_total:.2f} %",
                    "Rendement Annualisé": f"{rendement_annualise:.2f} %",
                    "Composition": df_portfolio
                }

                # Répartition par Secteur
                sectors = {}
                for ticker in actifs_list:
                    stock = yf.Ticker(ticker)
                    info = stock.info
                    sector = info.get("sector", "Inconnu")
                    sectors[ticker] = sector
                df_portfolio["Secteur"] = [sectors[ticker] for ticker in df_portfolio["Actif"]]
                sector_dist = df_portfolio.groupby("Secteur")["Valeur Actuelle"].sum().reset_index()
                fig_sector = px.pie(sector_dist, values="Valeur Actuelle", names="Secteur", title="Répartition par Secteur")
                st.plotly_chart(fig_sector)
                graphs["Répartition par Secteur"] = fig_sector

                # Évolution Temporelle
                df_historical = pd.DataFrame(historical_data)
                df_historical.index = df_historical.index.tz_localize(None)
                portfolio_value = pd.DataFrame(index=df_historical.index)
                for ticker in actifs_list:
                    initial_amount = df_portfolio[df_portfolio["Actif"] == ticker]["Montant Initial"].values[0]
                    shares = initial_amount / df_historical[ticker][0]
                    portfolio_value[ticker] = df_historical[ticker] * shares
                portfolio_value["Total"] = portfolio_value.sum(axis=1)
                fig_time = px.line(portfolio_value, x=portfolio_value.index, y="Total", title=f"Évolution du Portefeuille ({period})")
                st.plotly_chart(fig_time)
                graphs["Évolution du Portefeuille"] = fig_time

                # Risque du Portefeuille
                returns = df_historical.pct_change().dropna()
                portfolio_returns = (returns * (df_portfolio["Montant Initial"] / total_initial).values).sum(axis=1)
                volatility = portfolio_returns.std() * np.sqrt(252) * 100
                selected_data["Analyse de Portefeuille"]["Volatilité Annualisée"] = f"{volatility:.2f} %"

                # Corrélation entre Actifs
                correlation_matrix = returns.corr()
                fig_corr = px.imshow(correlation_matrix, text_auto=True, title="Matrice de Corrélation", color_continuous_scale="RdBu")
                st.plotly_chart(fig_corr)
                graphs["Matrice de Corrélation"] = correlation_matrix  # Ajout en tant que DataFrame pour le PDF

                # Répartition Initiale
                fig_pie = px.pie(df_portfolio, values="Montant Initial", names="Actif", title="Répartition Initiale")
                st.plotly_chart(fig_pie)
                graphs["Répartition Initiale"] = fig_pie

                # Frontière Efficiente et Portefeuille Optimal
                daily_returns = returns.dropna()
                mean_returns = daily_returns.mean() * 252
                cov_matrix = daily_returns.cov() * 252
                num_assets = len(actifs_list)
                num_portfolios = 10000

                np.random.seed(42)
                portfolio_returns = []
                portfolio_volatilities = []
                portfolio_weights = []

                for _ in range(num_portfolios):
                    weights = np.random.random(num_assets)
                    weights /= np.sum(weights)
                    portfolio_weights.append(weights)
                    port_return = np.sum(weights * mean_returns)
                    port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                    portfolio_returns.append(port_return * 100)
                    portfolio_volatilities.append(port_volatility * 100)

                portfolios = pd.DataFrame({
                    "Rendement (%)": portfolio_returns,
                    "Volatilité (%)": portfolio_volatilities
                })

                min_vol_idx = portfolios["Volatilité (%)"].idxmin()
                min_vol_weights = portfolio_weights[min_vol_idx]
                min_vol_return = portfolios["Rendement (%)"][min_vol_idx]
                min_vol_volatility = portfolios["Volatilité (%)"][min_vol_idx]

                target_returns = np.linspace(min(portfolio_returns), max(portfolio_returns), 50)
                efficient_volatilities = []

                def portfolio_volatility(weights, cov_matrix):
                    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

                def minimize_volatility(target_return, mean_returns, cov_matrix):
                    constraints = (
                        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                        {'type': 'eq', 'fun': lambda x: np.sum(x * mean_returns) - target_return}
                    )
                    bounds = tuple((0, 1) for _ in range(num_assets))
                    initial_weights = np.array([1/num_assets] * num_assets)
                    result = minimize(portfolio_volatility, initial_weights, args=(cov_matrix,),
                                    method='SLSQP', bounds=bounds, constraints=constraints)
                    return result.fun

                for target in target_returns:
                    efficient_vol = minimize_volatility(target / 100, mean_returns, cov_matrix) * 100
                    efficient_volatilities.append(efficient_vol)

                fig_eff = px.scatter(portfolios, x="Volatilité (%)", y="Rendement (%)", 
                                    title="Frontière Efficiente et Portefeuilles Simulés",
                                    labels={"Volatilité (%)": "Volatilité Annualisée (%)", "Rendement (%)": "Rendement Attendu (%)"},
                                    opacity=0.3)
                fig_eff.add_scatter(x=efficient_volatilities, y=target_returns, mode='lines', name='Frontière Efficiente', line=dict(color='green'))
                fig_eff.add_scatter(x=[min_vol_volatility], y=[min_vol_return], mode='markers', name='Portefeuille Variance Minimale', 
                                    marker=dict(color='red', size=10))
                st.plotly_chart(fig_eff)
                graphs["Frontière Efficiente"] = fig_eff

                min_vol_df = pd.DataFrame({
                    "Actif": actifs_list,
                    "Poids (%)": [w * 100 for w in min_vol_weights]
                })
                selected_data["Analyse de Portefeuille"]["Portefeuille Variance Minimale"] = {
                    "Composition": min_vol_df,
                    "Rendement Attendu": f"{min_vol_return:.2f} %",
                    "Volatilité": f"{min_vol_volatility:.2f} %"
                }

        if st.button("Générer le Rapport"):
            if selected_data:
                generate_pdf_report_with_graphics(user_info, selected_data, graphs)
                with open("rapport_personnalise.pdf", "rb") as file:
                    st.download_button("Télécharger le Rapport PDF", file, "rapport_personnalise.pdf")
            else:
                st.warning("Veuillez sélectionner au moins une option.")

# Footer
if st.session_state.authenticated:
    st.sidebar.write(f"Session de {st.session_state.user_info['prenom']} - Mars 2025")
