# Documentation Utilisateur : Tableau de Bord Financier

Bienvenue sur le **Tableau de Bord Financier**, une application interactive conçue pour vous aider à analyser vos finances, explorer des concepts financiers et générer des rapports personnalisés. Cette documentation vous guidera à travers les étapes pour utiliser l'application efficacement. Que vous soyez novice ou investisseur expérimenté, ce guide est là pour vous accompagner !

**Date de référence** : Mars 2025

---

## 1. Premiers Pas : Connexion
1. **Accéder à l'application** :
   - Ouvrez l'application dans votre navigateur.
   - Vous verrez une page de connexion intitulée "Bienvenue sur le Tableau de Bord Financier".

2. **S'authentifier** :
   - Remplissez les champs suivants :
     - **Nom** : Votre nom de famille.
     - **Prénom** : Votre prénom.
     - **Âge** : Votre âge (entre 1 et 120 ans).
     - **Email** : Optionnel, laissez vide si vous ne souhaitez pas le fournir.
   - Cliquez sur **Se connecter**.
   - Si un champ obligatoire est vide, un message d’erreur s’affichera.

3. **Choix de la section préférée** :
   - Après la connexion, vous serez accueilli(e) par votre prénom.
   - Sélectionnez une section dans la liste déroulante (par exemple, "Accueil (KPI)") pour commencer.
   - Cliquez sur **Confirmer** pour accéder au tableau de bord principal.

---

## 2. Navigation dans l’Application
- **Barre latérale** :
  - Votre prénom et âge sont affichés en haut.
  - Une astuce personnalisée apparaît selon votre âge (par exemple, "Investissez tôt !" pour les moins de 30 ans).
  - Utilisez le menu radio pour naviguer entre les sections :
    - Accueil (KPI)
    - Calculatrices Financières
    - Analyse de Portefeuille
    - Visualisation Boursière
    - Prédiction de Prix
    - Simulation Monte Carlo
    - Analyse de Sentiments
    - Quiz Financier
    - Rapport Personnalisé

---

## 3. Présentation des Sections

### 3.1. Accueil (KPI)
- **Objectif** : Suivre les performances de votre portefeuille avec des indicateurs clés.
- **Comment faire** :
  - Entrez vos actifs (ex. : "AAPL, MSFT") et les montants investis (ex. : "1000, 2000").
  - Choisissez une période (1 mois à 2 ans).
  - Consultez la valeur totale, le rendement et la volatilité moyenne.
  - Visualisez les prix historiques et les actualités financières.

### 3.2. Calculatrices Financières
- **Objectif** : Effectuer des calculs financiers simples.
- **Options** :
  - **Intérêts Composés** : Calculez la valeur future d’un investissement.
  - **Amortissement de Prêt** : Estimez vos paiements mensuels.
  - **VAN et TRI** : Analysez la rentabilité d’un projet avec des flux de trésorerie.
- **Astuce** : Entrez des nombres valides (ex. : "-5000, 2000" pour les flux).

### 3.3. Analyse de Portefeuille
- **Objectif** : Analyser en détail la performance et le risque de votre portefeuille.
- **Comment l’utiliser** :
  1. **Définir votre portefeuille** :
     - Entrez vos actifs (ex. : "AAPL, TSLA") dans le champ "Actifs".
     - Entrez les montants investis (ex. : "1500, 2500") dans "Montants investis".
     - Sélectionnez une période d’analyse (1 mois à 2 ans).
     - Assurez-vous que le nombre d’actifs correspond au nombre de montants, sinon un message d’erreur apparaîtra.
  2. **Résultats affichés** :
     - Valeur initiale et actuelle, rendement total et annualisé.
     - Répartition par secteur (graphique en camembert).
     - Évolution temporelle (graphique linéaire).
     - Volatilité annualisée.
     - Matrice de corrélation entre actifs.
     - Répartition initiale (graphique en camembert).
     - Frontière efficiente avec portefeuille à variance minimale (graphique de dispersion).
  3. **Exporter en PDF** :
     - **Important** : Pour inclure cette analyse dans un rapport PDF, vous **devez d’abord définir votre portefeuille et lancer l’analyse ici**. Sinon, aucune donnée ne sera disponible pour l’exportation.
     - Rendez-vous ensuite dans la section "Rapport Personnalisé", cochez "Analyse de Portefeuille", et générez le rapport.

### 3.4. Visualisation Boursière
- **Objectif** : Voir les prix historiques d’actifs.
- **Comment faire** :
  - Entrez les symboles (ex. : "AAPL, MSFT") et choisissez une période.
  - Un graphique interactif s’affiche.

### 3.5. Prédiction de Prix
- **Objectif** : Prévoir les prix futurs d’un actif.
- **Comment faire** :
  - Entrez un symbole (ex. : "AAPL") et choisissez un modèle (Régression Linéaire ou Prophet).
  - Visualisez les prédictions sur 30 jours avec une explication.

### 3.6. Simulation Monte Carlo
- **Objectif** : Simuler l’évolution d’un investissement.
- **Comment faire** :
  - Entrez un montant, une durée, et le nombre de trajectoires.
  - Choisissez entre "Basé sur un titre" (ex. : "AAPL") ou "Manuel" (rendement et volatilité personnalisés).
  - Consultez les trajectoires et les scénarios (pire, moyen, meilleur).

### 3.7. Analyse de Sentiments
- **Objectif** : Évaluer le sentiment d’un texte.
- **Comment faire** :
  - Entrez un texte (ex. : "Le marché est en hausse !").
  - Obtenez un score (positif, négatif, neutre).

### 3.8. Quiz Financier
- **Objectif** : Tester vos connaissances.
- **Comment faire** :
  - Choisissez un niveau (Facile, Moyen, Difficile).
  - Répondez aux questions dans un délai de 30 secondes chacune.
  - Consultez votre score final et recommencez si vous voulez.

### 3.9. Rapport Personnalisé
- **Objectif** : Générer un rapport PDF avec vos analyses.
- **Comment l’utiliser** :
  1. **Sélectionner les sections** :
     - Cochez les options que vous voulez inclure (ex. : "Analyse de Portefeuille", "Intérêts Composés").
     - Remplissez les champs demandés pour chaque option (montants, taux, etc.).
  2. **Générer le rapport** :
     - Cliquez sur "Générer le Rapport".
     - Téléchargez le fichier PDF via le bouton "Télécharger le Rapport PDF".
  3. **Notes importantes** :
     - Pour inclure l’**Analyse de Portefeuille**, vous devez d’abord la configurer dans la section correspondante (voir 3.3). Sinon, cette partie sera vide ou absente.
     - Les graphiques dans le PDF sont actuellement en **noir et blanc**. Une future mise à jour pourrait permettre une exportation en couleur, mais cette fonctionnalité n’est pas encore disponible.

---

## 4. Conseils Pratiques
- **Entrées valides** : Utilisez des virgules pour séparer les actifs ou montants (ex. : "AAPL, MSFT" ou "1000, 2000"). Évitez les lettres ou symboles non numériques dans les champs de montants.
- **Chargement** : Certaines sections (comme les données boursières) peuvent prendre quelques secondes à se charger. Attendez que le spinner disparaisse.
- **Erreurs** : Si une erreur apparaît (ex. : données indisponibles pour un actif), vérifiez votre saisie ou essayez un autre symbole.

---

## 5. Limitations Actuelles
- **Graphiques en PDF** : Les graphiques exportés dans les rapports sont en noir et blanc. Une amélioration future pourrait inclure des couleurs.
- **Données boursières** : Si un symbole est invalide ou les données sont indisponibles, l’application utilisera la valeur initiale ou affichera un avertissement.
- **Actualités** : Limitées à 60 articles par actif ou recherche, en français uniquement.

---

## 6. Besoin d’Aide ?
- Consultez les messages d’erreur ou d’avertissement affichés à l’écran pour corriger vos saisies.
- Pour des questions avancées, notez que cette version est éducative et analytique, sans support technique direct.

---

Merci d’utiliser le **Tableau de Bord Financier** ! Profitez de cette outil pour explorer, apprendre et gérer vos finances. Une fois votre portefeuille analysé, n’oubliez pas de générer un rapport pour conserver vos résultats !

---

### Focus sur l’Analyse de Portefeuille et l’Exportation PDF
Pour répondre à votre demande spécifique, la section "Analyse de Portefeuille" (3.3) et "Rapport Personnalisé" (3.9) insistent sur le fait que l’utilisateur doit configurer son portefeuille avant d’exporter un rapport. De plus, la limitation des graphiques en noir et blanc est clairement mentionnée avec une note sur une amélioration future, rendant cette information transparente.

Si vous souhaitez ajuster ou ajouter des sections à cette documentation (par exemple, des captures d’écran ou des exemples spécifiques), faites-le-moi savoir !
