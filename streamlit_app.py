import streamlit as st
import pandas as pd
import pulp as pu
from collections import defaultdict

def solve_with_pulp_integer(campaigns, total_leads, corpo_percent, min_share, budget_max):
    """
    Risolve il problema con PuLP, forzando x_i (lead per campagna i) a essere interi.
    
    Parametri:
    - campaigns: lista di dict {name, category, cost, revenue, net_profit}
    - total_leads: numero totale di lead da generare (intero)
    - corpo_percent: percentuale minima di lead 'corpo' (0..1)
    - min_share: percentuale minima di lead per OGNI campagna nella sua categoria (0..1)
    - budget_max: budget massimo disponibile (>= 0)

    Variabili: x_i >= 0, cat='Integer'

    Vincoli:
      1) somma(x_i) = total_leads
      2) somma(x_i in 'corpo') >= corpo_percent * total_leads
      3) per ogni categoria e per ogni campagna j in quella categoria:
         x_j >= min_share * (somma x nella cat)
      4) somma(cost_i * x_i) <= budget_max
    Obiettivo: max sum(net_profit_i * x_i)
    
    Nota: se min_share * (numero di campagne in cat) > 1, il problema diventa infeasible.
    """

    # Definiamo il problema (CBC solver) in massimizzazione
    prob = pu.LpProblem("MarketingCampaignOptimizationInteger", pu.LpMaximize)
    n = len(campaigns)

    # 1) Variabili di decisione (x_i intere)
    x = [pu.LpVariable(f"x_{i}", lowBound=0, cat="Integer") for i in range(n)]

    # 2) Funzione obiettivo (profitto totale)
    profit_expr = []
    for i, camp in enumerate(campaigns):
        profit_expr.append(camp["net_profit"] * x[i])
    prob += pu.lpSum(profit_expr), "Total_Profit"

    # 3) Vincoli

    # (a) Somma dei lead = total_leads
    prob += pu.lpSum(x) == total_leads, "Totale_lead"

    # (b) Somma dei lead 'corpo' >= corpo_percent * total_leads
    corpo_indices = [i for i, camp in enumerate(campaigns) if camp["category"] == "corpo"]
    if corpo_indices:
        prob += pu.lpSum([x[i] for i in corpo_indices]) >= corpo_percent * total_leads, "Minimo_corpo"

    # (c) min_share per OGNI campagna nella categoria
    cat_dict = defaultdict(list)
    for i, camp in enumerate(campaigns):
        cat_dict[camp["category"]].append(i)

    for category, indices in cat_dict.items():
        # somma dei lead nella categoria
        sum_in_cat = pu.lpSum([x[idx] for idx in indices])
        # Se c'è 1 sola campagna in questa categoria, min_share vale comunque
        # (ma se min_share>1, sarà infeasible).
        for j in indices:
            prob += x[j] >= min_share * sum_in_cat, f"MinShare_{category}_{j}"

    # (d) Vincolo di budget massimo: somma(cost_i * x_i) <= budget_max
    cost_expr = []
    for i, camp in enumerate(campaigns):
        cost_expr.append(camp["cost"] * x[i])
    prob += pu.lpSum(cost_expr) <= budget_max, "BudgetMax"

    # 4) Risolvi (CBC solver, silenzioso)
    prob.solve(pu.PULP_CBC_CMD(msg=0))
    status = pu.LpStatus[prob.status]

    # Se ottimale, recuperiamo i valori delle variabili
    if status == "Optimal":
        x_values = [pu.value(var) for var in x]
        total_profit = pu.value(prob.objective)
        return status, x_values, total_profit
    else:
        return status, None, None

def main():
    st.title("Ottimizzatore di Campagne con min_share per OGNI Campagna in Categoria")

    st.write("""
    - I lead sono **interi** (cat="Integer").
    - Vincolo: Ogni campagna riceve almeno `min_share` dei lead della propria categoria.
      (Attenzione a non superare 1/(num. campagne in cat).)
    - Altre funzioni: percentuale minima 'corpo', budget massimo, etc.
    """)

    st.subheader("1) Caricamento dei Dati")

    # Link al file di esempio
    mode = st.radio(
        "Come vuoi inserire i dati delle tue campagne? Carica un CSV come [questo file di esempio](https://drive.google.com/file/d/1vfp_gd6ivHsVpxffn_seAB11qCy9m0bP/view?usp=sharing)",
        ["Carica CSV", "Inserimento manuale"]
    )
    campaigns = []

    if mode == "Carica CSV":
        uploaded_file = st.file_uploader("Seleziona il tuo CSV", type=["csv"])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            
            st.write("**Anteprima del CSV caricato:**")
            st.dataframe(df.head())

            # Normalizziamo i nomi delle colonne
            df.columns = [c.lower().strip() for c in df.columns]
            required_cols = ["nome campagna", "categoria campagna", "costo per lead", "ricavo per lead"]
            if all(col in df.columns for col in required_cols):
                for idx, row in df.iterrows():
                    name = str(row["nome campagna"]).strip()
                    category = str(row["categoria campagna"]).lower().strip()
                    cost = float(row["costo per lead"])
                    revenue = float(row["ricavo per lead"])
                    net_profit = revenue - cost

                    campaigns.append({
                        "name": name,
                        "category": category,
                        "cost": cost,
                        "revenue": revenue,
                        "net_profit": net_profit
                    })
            else:
                st.error(f"Le colonne attese sono: {required_cols}. Controlla il tuo CSV.")
    else:
        n = st.number_input("Numero di campagne (2..10):", min_value=2, max_value=10, value=2, step=1)
        for i in range(n):
            st.markdown(f"**Campagna #{i+1}**")
            name = st.text_input(f"Nome campagna #{i+1}", key=f"name_{i}")
            category = st.selectbox(f"Categoria #{i+1}", ["laser", "corpo"], key=f"cat_{i}")
            cost = st.number_input(f"Costo per lead #{i+1}", min_value=0.0, value=0.0, step=1.0, key=f"cost_{i}")
            revenue = st.number_input(f"Ricavo per lead #{i+1}", min_value=0.0, value=0.0, step=1.0, key=f"rev_{i}")
            
            net_profit = revenue - cost
            campaigns.append({
                "name": name if name else f"Camp_{i+1}",
                "category": category,
                "cost": cost,
                "revenue": revenue,
                "net_profit": net_profit
            })

    st.write("---")

    if campaigns:
        st.subheader("2) Parametri Globali")

        total_leads = st.number_input("Totale dei lead da produrre (intero):", min_value=1, value=10000, step=1)
        corpo_percent = st.slider(
            "Percentuale minima di lead 'corpo' (0% = 0.0, 100% = 1.0):", 
            min_value=0.0, max_value=1.0, value=0.33, step=0.01
        )
        min_share = st.slider(
            "Percentuale minima di lead per OGNI campagna nella propria categoria",
            min_value=0.0, max_value=1.0, value=0.2, step=0.01
        )
        budget_max = st.number_input(
            "Budget massimo (EUR) che non vuoi superare:", 
            min_value=0.0, 
            value=50000.0, 
            step=100.0
        )

        if st.button("Esegui Ottimizzazione"):
            status, x_values, total_profit = solve_with_pulp_integer(
                campaigns,
                int(total_leads),
                corpo_percent,
                min_share,
                budget_max
            )

            if status != "Optimal":
                st.error(f"Soluzione non ottimale o infeasible. Stato solver: {status}")
                return

            # Prepariamo la tabella di output
            results = []
            cost_total_sum = 0
            revenue_total_sum = 0
            profit_total_sum = 0

            for i, camp in enumerate(campaigns):
                leads_float = x_values[i] or 0.0
                leads = int(round(leads_float))

                cost_t = camp["cost"] * leads
                revenue_t = camp["revenue"] * leads
                margin_t = camp["net_profit"] * leads

                cost_total_sum += cost_t
                revenue_total_sum += revenue_t
                profit_total_sum += margin_t

                results.append({
                    "Campagna": camp["name"],
                    "Categoria": camp["category"],
                    "Leads": leads,
                    "Costo Tot": int(round(cost_t)),
                    "Ricavo Tot": int(round(revenue_t)),
                    "Margine": int(round(margin_t))
                })
            
            st.write("**Assegnazione Campagne**")
            st.table(results)

            # Calcoliamo quanti lead 'corpo' totali
            corpo_leads_used = sum(r["Leads"] for r in results if r["Categoria"] == "corpo")

            st.write("**Riepilogo**")
            st.write(f"Totale lead: {int(total_leads)}")
            st.write(f"Lead 'corpo': {corpo_leads_used} (≥ {int(round(corpo_percent*100))}% del totale)")
            st.write(f"Lead 'laser': {int(total_leads - corpo_leads_used)}")

            st.write(f"Costo totale: {int(round(cost_total_sum))} (non supera {int(budget_max)}€)")
            st.write(f"Ricavo totale: {int(round(revenue_total_sum))}")
            st.write(f"Profitto totale: {int(round(profit_total_sum))}")

            st.info("""
                **Attenzione**: se hai 2 o più campagne in una categoria e `min_share` * (numero di campagne in quella categoria) > 1,
                il problema risulterà infeasible.
            """)
        else:
            st.info("Imposta i parametri e clicca su 'Esegui Ottimizzazione'.")
    else:
        st.info("Carica un CSV valido o inserisci le campagne manualmente.")

if __name__ == "__main__":
    main()
