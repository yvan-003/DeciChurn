from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


st.set_page_config(page_title="DeciChurn", layout="wide")


def inject_css() -> None:
    # Thème  (fond, cartes, onglets) 
    st.markdown(
        """
        <style>
        :root {
            --card-bg: rgba(255, 255, 255, 0.04);
            --card-border: rgba(255, 255, 255, 0.08);
            --accent-1: #33a1fd;
            --accent-2: #8a6cff;
        }
        [data-testid="stAppViewContainer"] {
            background: radial-gradient(120% 120% at 0% 0%, #102347 0, #0b0f1f 40%, #060910 100%);
            color: #e9edf5;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1250px;
        }
        .hero {
            background: linear-gradient(135deg, rgba(51,161,253,0.18), rgba(138,108,255,0.14));
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 20px;
            padding: 1.6rem 1.8rem;
            box-shadow: 0 24px 70px rgba(0,0,0,0.35);
        }
        .hero .eyebrow {
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-size: 0.9rem;
            color: #89c4ff;
        }
        .hero h1 {
            margin: 0.2rem 0 0.3rem 0;
            font-size: 2.2rem;
        }
        .hero p {
            margin: 0;
            color: #dfe6f3;
        }
        .pill {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background: rgba(255,255,255,0.05);
            padding: 0.35rem 0.8rem;
            border-radius: 999px;
            border: 1px solid rgba(255, 255, 255, 0.08);
            font-size: 0.95rem;
        }
        div[data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 14px;
            padding: 1rem 1rem;
            box-shadow: 0 12px 35px rgba(0,0,0,0.28);
        }
        .stTabs [role="tab"] {
            background: rgba(255,255,255,0.05);
            border: 1px solid transparent;
            margin-right: 6px;
            padding: 0.65rem 1rem;
            border-radius: 12px;
        }
        .stTabs [role="tab"][aria-selected="true"] {
            background: linear-gradient(120deg, var(--accent-1), var(--accent-2));
            color: #0b0f1a;
            font-weight: 700;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    # Lecture du CSV local et coercition de TotalCharges en numérique
    df = pd.read_csv(path)
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    return df


def compute_kpis(df: pd.DataFrame) -> dict:
    # Calcul des indicateurs affichés dans la rangée KPI
    total_clients = len(df)
    churn_rate = df["Churn"].eq("Yes").mean() if "Churn" in df.columns else float("nan")
    missing_totalcharges = int(df["TotalCharges"].isna().sum()) if "TotalCharges" in df.columns else 0
    arpu = df["MonthlyCharges"].mean() if "MonthlyCharges" in df.columns else float("nan")

    churn_high_impact = float("nan")
    high_impact_threshold = float("nan")
    if {"MonthlyCharges", "Churn"} <= set(df.columns):
        high_impact_threshold = df["MonthlyCharges"].median()
        hi = df[df["MonthlyCharges"] > high_impact_threshold]
        if len(hi) > 0:
            churn_high_impact = hi["Churn"].eq("Yes").mean()

    return {
        "total_clients": total_clients,
        "churn_rate": churn_rate,
        "missing_totalcharges": missing_totalcharges,
        "arpu": arpu,
        "churn_high_impact": churn_high_impact,
        "high_impact_threshold": high_impact_threshold,
    }


def render_churn_by_contract(df: pd.DataFrame) -> None:
    # Bar chart du churn moyen par type de contrat
    if {"Contract", "Churn"} <= set(df.columns):
        data = (
            df.assign(churn_flag=df["Churn"].eq("Yes"))
            .groupby("Contract")["churn_flag"]
            .mean()
            .reset_index()
            .rename(columns={"churn_flag": "churn_pct"})
        )
        data["churn_pct"] *= 100

        chart = (
            alt.Chart(data)
            .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
            .encode(
                x=alt.X("Contract:N", title="Type de contrat", sort="-y"),
                y=alt.Y("churn_pct:Q", title="Taux de churn (%)"),
                color=alt.Color("Contract:N", legend=None),
                tooltip=["Contract", alt.Tooltip("churn_pct:Q", format=".1f", title="Churn (%)")],
            )
            .properties(height=320)
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("Colonnes Contract / Churn absentes.")


def render_charges_vs_tenure(df: pd.DataFrame) -> None:
    # Scatter pour voir la relation ancienneté / charges, coloré par churn
    if {"MonthlyCharges", "tenure"} <= set(df.columns):
        plot_df = df.sample(n=min(2000, len(df)), random_state=42)

        encoding = {
            "x": alt.X("tenure:Q", title="Ancienneté (mois)"),
            "y": alt.Y("MonthlyCharges:Q", title="Facture mensuelle ($)"),
            "tooltip": [c for c in ["customerID", "tenure", "MonthlyCharges", "Contract", "Churn"] if c in df.columns],
        }

        if "Churn" in df.columns:
            encoding["color"] = alt.Color(
                "Churn:N",
                title="Churn",
                scale=alt.Scale(domain=["No", "Yes"], range=["#6ee7b7", "#f97066"]),
            )
        else:
            encoding["color"] = alt.value("#6ee7b7")

        chart = (
            alt.Chart(plot_df)
            .mark_circle(size=38, opacity=0.55)
            .encode(**encoding)
            .properties(height=320)
            .interactive()
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("Colonnes MonthlyCharges / tenure absentes.")


def build_pipeline(numeric_cols: list[str], categorical_cols: list[str]) -> Pipeline:
    # Préprocesseur (imputation + one-hot) combiné à une régression logistique
    numeric_pipe = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
    )

    model = LogisticRegression(max_iter=2000)
    return Pipeline(steps=[("prep", preprocessor), ("model", model)])


def get_feature_names(pipe: Pipeline) -> list[str]:
    # Récupération des noms de features après transformation pour affichage des coefficients
    prep: ColumnTransformer = pipe.named_steps["prep"]
    feature_names: list[str] = []

    num_cols = prep.transformers_[0][2]
    feature_names.extend(list(num_cols))

    cat_pipe = prep.transformers_[1][1]
    ohe: OneHotEncoder = cat_pipe.named_steps["onehot"]
    cat_cols = prep.transformers_[1][2]
    feature_names.extend(list(ohe.get_feature_names_out(cat_cols)))

    return feature_names


@st.cache_data
def train_and_score(df: pd.DataFrame, test_size: float, seed: int):
    """Train logistic regression with preprocessing; return predictions and metadata."""
    # Validation de la présence/valeurs de la cible
    if "Churn" not in df.columns:
        raise ValueError("Colonne 'Churn' absente.")

    y = df["Churn"].map({"No": 0, "Yes": 1})
    if y.isna().any():
        raise ValueError("Valeurs inattendues dans 'Churn' (attendu: 'No'/'Yes').")

    # Constitution des features (exclusion id et cible)
    drop_cols = {"Churn"}
    if "customerID" in df.columns:
        drop_cols.add("customerID")

    X = df.drop(columns=list(drop_cols), errors="ignore").copy()

    # Détection auto numérique/catégoriel + split stratifié
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    pipe = build_pipeline(numeric_cols, categorical_cols)
    pipe.fit(X_train, y_train)

    # Probas sur test et sur tout le jeu (pour ciblage en masse)
    proba_test = pipe.predict_proba(X_test)[:, 1]
    proba_all = pipe.predict_proba(X)[:, 1]

    feature_names = get_feature_names(pipe)
    coefs = pipe.named_steps["model"].coef_.ravel()
    importance = (
        pd.DataFrame({"feature": feature_names, "coef": coefs})
        .assign(abs_coef=lambda d: d["coef"].abs())
        .sort_values("abs_coef", ascending=False)
        .reset_index(drop=True)
    )

    meta = {
        "n_train": len(X_train),
        "n_test": len(X_test),
        "base_rate": float(y.mean()),
    }

    return pipe, X_test, y_test, proba_test, X, proba_all, meta, importance


def decision_metrics(y_true: np.ndarray, proba: np.ndarray, threshold: float):
    # Métriques classiques + matrice confusion pour un seuil donné
    y_pred = (proba >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def optimize_threshold(y_true: np.ndarray, proba: np.ndarray, cost_fp: float, cost_fn: float):
    """Grid search on thresholds to minimize business cost built from FP/FN."""
    # Balaye des seuils de 0.05 à 0.95 par pas de 0.01
    thresholds = np.linspace(0.05, 0.95, 91)
    rows = []
    for t in thresholds:
        m = decision_metrics(y_true, proba, float(t))
        total_cost = m["fp"] * cost_fp + m["fn"] * cost_fn
        rows.append(
            {
                "threshold": float(t),
                "cost": float(total_cost),
                "precision": m["precision"],
                "recall": m["recall"],
                "fp": m["fp"],
                "fn": m["fn"],
                "tp": m["tp"],
                "tn": m["tn"],
            }
        )
    res = pd.DataFrame(rows).sort_values("cost", ascending=True).reset_index(drop=True)
    best = res.iloc[0].to_dict()
    return best, res


def recommend_actions(row: pd.Series, proba: float) -> list[str]:
    """Return a small action playbook based on profile and churn probability."""
    # Mini playbook conditionnel basé sur proba churn et attributs clés
    actions = []

    if proba >= 0.70:
        actions.append("Priorité haute : contact humain (appel) sous 48h.")
    elif proba >= 0.50:
        actions.append("Priorité moyenne : email + offre ciblée.")
    else:
        actions.append("Priorité faible : nurturing / suivi léger.")

    if row.get("Contract", None) == "Month-to-month":
        actions.append("Proposer migration vers 1 an / 2 ans avec rabais.")

    mc = row.get("MonthlyCharges", np.nan)
    if pd.notna(mc) and mc >= 80:
        actions.append("Offre : réduction temporaire / bundle pour baisser la facture.")

    tenure = row.get("tenure", np.nan)
    if pd.notna(tenure) and tenure <= 6:
        actions.append("Onboarding renforcé + check-in satisfaction (clients récents).")

    if row.get("InternetService", None) == "Fiber optic":
        actions.append("Vérifier qualité service + geste commercial si besoin.")

    if row.get("TechSupport", None) == "No":
        actions.append("Proposer TechSupport (gratuit 1 mois) pour réduire frictions.")

    return actions[:5]


DATA_PATH = Path(__file__).parent / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
df = load_data(DATA_PATH)

inject_css()

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">Customer Success</div>
        <h1>DeciChurn — Customer Churn Decision Platform</h1>
        <p>Suivez le churn, ciblez la rétention et explorez vos clients en un clin d'œil.</p>
        <div style="display:flex; gap:8px; flex-wrap:wrap; margin-top:0.9rem;">
            <span class="pill">KPI</span>
            <span class="pill">Exploration</span>
            <span class="pill">Modèle</span>
            <span class="pill">Seuil optimal</span>
            <span class="pill">Budget</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

k = compute_kpis(df)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Clients", f"{k['total_clients']:,}".replace(",", " "))
c2.metric("Taux de churn", f"{k['churn_rate']*100:.1f}%" if pd.notna(k["churn_rate"]) else "N/A")
c3.metric("ARPU (mensuel moyen)", f"${k['arpu']:.2f}" if pd.notna(k["arpu"]) else "N/A")
hi_display = f"{k['churn_high_impact']*100:.1f}%" if pd.notna(k["churn_high_impact"]) else "N/A"
hi_help = f"Seuil ≈ ${k['high_impact_threshold']:.2f} (médiane)" if pd.notna(k["high_impact_threshold"]) else ""
c4.metric("Churn haut impact", hi_display, help=hi_help)

st.caption(f"Qualité données : TotalCharges manquants = {k['missing_totalcharges']:,}".replace(",", " "))

tab_overview, tab_target, tab_model, tab_data = st.tabs(
    ["Vue globale", "Ciblage (règle)", "Modèle & Décision", "Données brutes"]
)

with tab_overview:
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("**Churn par type de contrat**")
        render_churn_by_contract(df)
    with col_r:
        st.markdown("**Charges mensuelles vs ancienneté**")
        render_charges_vs_tenure(df)

with tab_target:
    st.markdown("#### Ciblage rétention — règle simple (baseline)")
    st.caption("Contrat mensuel, faible ancienneté, facture élevée.")

    col_a, col_b = st.columns(2)
    tenure_max = col_a.slider("Ancienneté max (mois)", 0, 72, 12, key="tenure_rule")
    monthly_charge_min = col_b.slider("Facture mensuelle min ($)", 0, 200, 70, key="charge_rule")

    filtered = df.copy()
    if "Contract" in filtered.columns:
        filtered = filtered[filtered["Contract"] == "Month-to-month"]
    if "tenure" in filtered.columns:
        filtered = filtered[filtered["tenure"] <= tenure_max]
    if "MonthlyCharges" in filtered.columns:
        filtered = filtered[filtered["MonthlyCharges"] >= monthly_charge_min]

    global_churn = df["Churn"].eq("Yes").mean()
    segment_churn = filtered["Churn"].eq("Yes").mean() if len(filtered) > 0 else float("nan")

    m1, m2, m3 = st.columns(3)
    m1.metric("Churn global", f"{global_churn*100:.1f}%")
    m2.metric("Churn segment", f"{segment_churn*100:.1f}%" if pd.notna(segment_churn) else "N/A")
    m3.metric("Lift", f"{segment_churn/global_churn:.2f}x" if pd.notna(segment_churn) and global_churn > 0 else "N/A")

    st.metric("Clients à cibler (règle)", f"{len(filtered):,}".replace(",", " "))
    cols_to_show = [c for c in ["customerID", "tenure", "MonthlyCharges", "Contract", "Churn"] if c in filtered.columns]
    if len(filtered) > 0 and cols_to_show:
        st.dataframe(filtered[cols_to_show].sort_values("tenure").head(50), use_container_width=True)
    else:
        st.info("Aucun client ne correspond aux filtres.")

with tab_model:
    st.markdown("#### Modèle & Décision — seuil optimal + simulateur budget")

    col_a, col_b = st.columns(2)
    test_size = col_a.slider("Taille du test (%)", 10, 40, 20) / 100
    seed = col_b.number_input("Random seed", min_value=0, max_value=9999, value=42, step=1)

    st.divider()
    st.markdown("**Hypothèses de coût** (pour optimiser le seuil)")
    col1, col2 = st.columns(2)
    cost_fp = col1.number_input("Coût faux positif (contacter un non-churn)", min_value=0.0, value=2.0, step=0.5)
    cost_fn = col2.number_input("Coût faux négatif (laisser partir un churn)", min_value=0.0, value=20.0, step=1.0)

    try:
        pipe, X_test, y_test, proba_test, X_all, proba_all, meta, importance = train_and_score(df, test_size, int(seed))
    except Exception as e:
        st.error(f"Erreur entraînement : {e}")
    else:
        best, grid = optimize_threshold(y_test.to_numpy(), proba_test, float(cost_fp), float(cost_fn))

        st.markdown("### 1) Seuil optimal (minimise le coût)")
        cA, cB, cC, cD = st.columns(4)
        cA.metric("Seuil optimal", f"{best['threshold']:.2f}")
        cB.metric("Coût minimal (test)", f"${best['cost']:,.2f}".replace(",", " "))
        cC.metric("Precision @opt", f"{best['precision']*100:.1f}%")
        cD.metric("Recall @opt", f"{best['recall']*100:.1f}%")

        cost_chart = (
            alt.Chart(grid)
            .mark_line()
            .encode(
                x=alt.X("threshold:Q", title="Seuil"),
                y=alt.Y("cost:Q", title="Coût (FP*cost_fp + FN*cost_fn)"),
                tooltip=[
                    alt.Tooltip("threshold:Q", format=".2f"),
                    alt.Tooltip("cost:Q", format=".2f"),
                    alt.Tooltip("precision:Q", format=".2f"),
                    alt.Tooltip("recall:Q", format=".2f"),
                ],
            )
            .properties(height=260)
        )
        st.altair_chart(cost_chart, use_container_width=True)

        st.caption(f"Base rate churn ≈ {meta['base_rate']*100:.1f}% — Train: {meta['n_train']}, Test: {meta['n_test']}")

        st.divider()

        st.markdown("### 2) Choix du seuil (optimal ou manuel)")
        use_opt = st.toggle("Utiliser automatiquement le seuil optimal", value=True)
        if use_opt:
            threshold = float(best["threshold"])
        else:
            threshold = st.slider("Seuil manuel", 0.05, 0.95, float(best["threshold"]))

        m = decision_metrics(y_test.to_numpy(), proba_test, threshold)
        total_cost = m["fp"] * cost_fp + m["fn"] * cost_fn

        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Accuracy", f"{m['accuracy']*100:.1f}%")
        k2.metric("Precision", f"{m['precision']*100:.1f}%")
        k3.metric("Recall", f"{m['recall']*100:.1f}%")
        k4.metric("FN", f"{m['fn']}")
        k5.metric("Coût estimé (test)", f"${total_cost:,.2f}".replace(",", " "))

        cm_df = pd.DataFrame(
            {"Prédit Non-churn": [m["tn"], m["fn"]], "Prédit Churn": [m["fp"], m["tp"]]},
            index=["Réel Non-churn", "Réel Churn"],
        )
        st.markdown("**Matrice de confusion**")
        st.dataframe(cm_df, use_container_width=True)

        st.divider()

        st.markdown("### 3) Explicabilité (top facteurs)")
        top_n = st.slider("Nombre de facteurs", 5, 30, 12)
        top_imp = importance.head(top_n).copy()
        top_imp["direction"] = np.where(top_imp["coef"] >= 0, "↑ risque churn", "↓ réduit churn")
        imp_chart = (
            alt.Chart(top_imp)
            .mark_bar(cornerRadius=6)
            .encode(
                y=alt.Y("feature:N", sort="-x", title="Feature"),
                x=alt.X("abs_coef:Q", title="Importance (|coef|)"),
                tooltip=[
                    alt.Tooltip("feature:N", title="Feature"),
                    alt.Tooltip("coef:Q", format=".4f", title="Coef"),
                    alt.Tooltip("direction:N", title="Effet"),
                ],
            )
            .properties(height=360)
        )
        st.altair_chart(imp_chart, use_container_width=True)

        st.divider()

        st.markdown("### 4) Simulateur budget : j’ai X contacts, qui cibler ?")
        st.caption("On trie les clients par probabilité de churn décroissante, puis on prend le top-N.")

        max_contacts = st.slider("Capacité (nombre de contacts)", 0, min(2000, len(df)), 200)
        contact_cost = st.number_input("Coût par contact ($)", min_value=0.0, value=1.0, step=0.5)
        retention_success = st.slider("Taux de succès rétention (si contact)", 0.0, 1.0, 0.20)

        out = df.copy()
        out["churn_proba"] = proba_all

        strategy = st.radio("Stratégie", ["Top-N (budget)", "Seuil (threshold)"], horizontal=True)

        if strategy == "Seuil (threshold)":
            targets = out[out["churn_proba"] >= threshold].sort_values("churn_proba", ascending=False)
            targets = targets.head(max_contacts)
        else:
            targets = out.sort_values("churn_proba", ascending=False).head(max_contacts)

        expected_churn_targets = float(targets["churn_proba"].sum()) if len(targets) > 0 else 0.0
        expected_saved = expected_churn_targets * float(retention_success)
        budget_total = float(max_contacts) * float(contact_cost)

        expected_value_saved = expected_saved * float(cost_fn)
        net_value = expected_value_saved - budget_total

        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Ciblés", f"{len(targets):,}".replace(",", " "))
        b2.metric("Churn attendu (ciblés)", f"{expected_churn_targets:.1f}")
        b3.metric("Churn évité (attendu)", f"{expected_saved:.1f}")
        b4.metric("Valeur nette (proxy)", f"${net_value:,.2f}".replace(",", " "))

        st.caption(
            "Interprétation : 'Valeur nette' = (churn évité attendu × coût FN) − (budget contacts). "
            "C’est une approximation utile pour simuler des scénarios."
        )

        show_cols = [c for c in ["customerID", "churn_proba", "MonthlyCharges", "tenure", "Contract", "InternetService", "TechSupport", "Churn"] if c in targets.columns]
        st.dataframe(targets[show_cols].head(50), use_container_width=True)

        csv = targets[show_cols].to_csv(index=False).encode("utf-8")
        st.download_button(
            "Télécharger la liste ciblée (CSV)",
            data=csv,
            file_name="decichurn_targets_budget.csv",
            mime="text/csv",
        )

        st.divider()

        st.markdown("### 5) Pourquoi ce client ? + actions")
        if len(targets) == 0:
            st.info("Aucun client sélectionné (budget 0 ou filtres).")
        else:
            if "customerID" in targets.columns:
                chosen_id = st.selectbox("Choisir un client (customerID)", targets["customerID"].head(200).tolist())
                chosen_row = targets[targets["customerID"] == chosen_id].iloc[0]
            else:
                chosen_idx = st.selectbox("Choisir un client (index)", targets.index[:200].tolist())
                chosen_row = targets.loc[chosen_idx]

            proba = float(chosen_row["churn_proba"])
            st.metric("Probabilité churn (client)", f"{proba*100:.1f}%")
            actions = recommend_actions(chosen_row, proba)

            st.markdown("**Actions recommandées**")
            for a in actions:
                st.write(f"- {a}")

            profile_cols = [c for c in ["tenure", "MonthlyCharges", "Contract", "InternetService", "TechSupport", "PaymentMethod", "PaperlessBilling"] if c in chosen_row.index]
            st.markdown("**Profil (extrait)**")
            st.dataframe(pd.DataFrame({"Valeur": chosen_row[profile_cols]}), use_container_width=True)

with tab_data:
    st.markdown("#### Aperçu des données")
    st.caption("50 premières lignes du fichier source.")
    st.dataframe(df.head(50), use_container_width=True, height=520)
