# DeciChurn — Customer Churn Decision Platform

Application Streamlit d’aide à la décision pour la rétention client :
- KPI (Clients, Churn, ARPU, Churn haut impact)
- Segmentation règle métier (baseline)
- Modèle prédictif (Logistic Regression) + seuil + coûts FP/FN
- Optimisation automatique du seuil
- Simulateur budget (Top-N / Threshold) + export CSV
- Explicabilité + recommandations d’actions

## Lancer en local
```bash
pip install -r requirements.txt
streamlit run app.py