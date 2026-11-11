# =============================================================================
# 1. IMPORTATION DES BIBLIOTHÈQUES
# =============================================================================
import matplotlib.pyplot as plt
import numpy as np

# =============================================================================
# 2. DÉFINITION DES PARAMÈTRES ET DEMI-ÉTENDUES
# =============================================================================
c = 4.18         # Capacité thermique massique de l'eau (en J·K^-1·g^-1)
me = 150.0       # Masse d'eau initialement présente dans le calorimètre (en g)
Ti = 20.0        # Température initiale de l'eau dans le calorimètre (en °C)
Ccal = 100.0     # Capacité thermique du calorimètre issue de la Partie 1 (en J·K^-1)
M = 80.0         # Masse molaire du nitrate d'ammonium NH4NO3 (en g·mol^-1)
DrH_tab = 25670  # Enthalpie standard de dissolution tabulée à 25°C (en J·mol^-1)

# Saisie des variables expérimentales
m = np.array([3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0])
T = np.array([18.9, 17.6, 16.2, 15.3, 14.1, 13.1, 11.8])

# Demi-étendues des instruments de mesure (loi uniforme)
a_T = 0.2        # Demi-étendue sur une lecture de température (en °C ou K)
a_m = 0.1        # Demi-étendue sur une pesée (en g)
u_Ccal = 15.0    # Incertitude-type sur Ccal déterminée en Partie 2 (en J·K^-1)

# =============================================================================
# 3. GRANDEURS EXPÉRIMENTALES DE RÉFÉRENCE
# =============================================================================
mu = Ccal / c          # Masse en eau du calorimètre (en g)
DeltaT = T - Ti        # Variation de température mesurée (en °C ou K)
nb_pts = len(T)        # Nombre de points expérimentaux (7 points)

# =============================================================================
# 4. SIMULATION DE MONTE-CARLO (N RÉGRESSIONS LINÉAIRES)
# =============================================================================
N = 50000
CD, OO, DrH = [], [], []

for i in range(N):
    # Tirage indépendant pour CHAQUE point expérimental (taille = nb_pts)
    T_simul = T + np.random.uniform(-a_T, a_T, nb_pts)
    m_simul = m + np.random.uniform(-a_m, a_m, nb_pts)
    
    # Tirage unique pour les grandeurs initiales (température et masse d'eau initiales)
    Ti_simul = Ti + np.random.uniform(-a_T, a_T)
    me_simul = me + np.random.uniform(-a_m, a_m)
    Ccal_simul = Ccal + np.random.normal(0, u_Ccal)
    
    DeltaT_simul = T_simul - Ti_simul
    mu_simul = Ccal_simul / c
    
    # Régression linéaire : 1/DeltaT = CD * (1/m) + OO
    p = np.polyfit(1 / m_simul, 1 / DeltaT_simul, 1)
    
    CD.append(p[0])
    OO.append(p[1])
    DrH.append(-((me_simul + mu_simul) * c * M) / p[0])

# Conversion en tableaux NumPy pour faciliter les calculs statistiques
CD = np.array(CD)
OO = np.array(OO)
DrH = np.array(DrH)

# =============================================================================
# 5. AFFICHAGE DES RÉSULTATS ET DU Z-SCORE
# =============================================================================
CD_moy, u_CD = np.mean(CD), np.std(CD, ddof=1)
OO_moy, u_OO = np.mean(OO), np.std(OO, ddof=1)
DrH_moy, u_DrH = np.mean(DrH), np.std(DrH, ddof=1)
z_score = abs(DrH_moy - DrH_tab) / u_DrH

print("=== RÉGRESSION LINÉAIRE AVEC SIMULATIONS MONTE-CARLO ===")
print(f"Coefficient directeur : CD = {CD_moy:.3f} g·K^-1  avec u(CD) = {u_CD:.3f} g·K^-1")
print(f"Ordonnée à l'origine  : OO = {OO_moy:.4f} K^-1    avec u(OO) = {u_OO:.4f} K^-1\n")

print("=== ENTHALPIE STANDARD DE DISSOLUTION ===")
print(f"DrH° = ({DrH_moy:.0f} avec u(DrH°) : {u_DrH/1000:.0f}) kJ·mol^-1")

print("=== COMPARAISON AVEC LA VALEUR TABULÉE (25 670 J·mol^-1) ===")
print(f"Écart normalisé (z-score) = {z_score:.2f}")

# =============================================================================
# 6. REPRÉSENTATION GRAPHIQUE (RÉGRESSION + HISTOGRAMME)
# =============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# --- Graphique 1 : Linéarisation 1/DeltaT = f(1/m) ---
X_exp = 1 / m
Y_exp = 1 / DeltaT
X_mod = np.linspace(min(X_exp) * 0.9, max(X_exp) * 1.05, 100)
Y_mod = CD_moy * X_mod + OO_moy

ax1.plot(X_exp, Y_exp, 'bo', markersize=6, label="Points expérimentaux")
ax1.plot(X_mod, Y_mod, 'r-', linewidth=1.8, 
         label=f"Modèle moyen : $y = {CD_moy:.2f}\\,x {OO_moy:+.3f}$")
ax1.set_xlabel("$1 / m$ ($\\mathrm{g^{-1}}$)")
ax1.set_ylabel("$1 / \\Delta T$ ($\\mathrm{K^{-1}}$)")
ax1.set_title("Linéarisation du bilan enthalpique")
ax1.legend()
ax1.grid(True, alpha=0.4)

# --- Graphique 2 : Distribution de Monte-Carlo pour DrH° ---
ax2.hist(DrH / 1000, bins=60, density=True, color='mediumseagreen', edgecolor='black', alpha=0.7)
ax2.axvline(DrH_moy / 1000, color='red', linestyle='--', linewidth=2, 
            label=f"Moyenne : {DrH_moy/1000:.2f} kJ·mol$^{{-1}}$")
ax2.axvline(DrH_tab / 1000, color='blue', linestyle='-', linewidth=2, 
            label=f"Valeur tabulée : {DrH_tab/1000:.2f} kJ·mol$^{{-1}}$")
ax2.set_xlabel("$\\Delta_{\\mathrm{r}}H^\\circ$ ($\\mathrm{kJ\\cdot mol^{-1}}$)")
ax2.set_ylabel("Densité de probabilité")
ax2.set_title(f"Distribution de Monte-Carlo (N = {N}, z_score = {z_score:.2f})")
ax2.legend()
ax2.grid(True, alpha=0.4)

plt.tight_layout()
plt.show()
