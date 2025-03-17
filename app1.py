import streamlit as st

# Titre de l'application
st.title("Calculatrice d'Intérêts Simples")

# Entrées utilisateur
principal = st.number_input("Montant initial (principal)", min_value=0.0, value=1000.0)
taux = st.number_input("Taux d'intérêt annuel (%)", min_value=0.0, value=5.0)
temps = st.number_input("Durée (en années)", min_value=0.0, value=1.0)

# Calcul des intérêts simples
interet = principal * (taux / 100) * temps
montant_total = principal + interet

# Bouton pour calculer
if st.button("Calculer"):
    st.write(f"Intérêts gagnés : {interet:.2f}")
    st.write(f"Montant total : {montant_total:.2f}")

# Instructions pour exécuter
st.write("Pour exécuter : sauvegardez ce code dans un fichier (ex: app.py) et lancez 'streamlit run app.py' dans le terminal.")