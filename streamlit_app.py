import streamlit as st
import pandas as pd
import pulp as pu
from collections import defaultdict
import io

# Per PDF (se poi volessi anche l'esportazione)
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors

def solve_mip(
    campaigns, 
    total_leads, 
    corpo_percent, 
    min_share, 
    budget_max
):
    """
    Risolve il problema con PuLP, cat='Integer', 
    con i vincoli:
      1) somma(x_i) = total_leads
      2) somma(x_i in 'corpo') >= corpo_percent * total_leads
      3) per ogni categoria, x_j >= min_share * somma(x in cat)
      4) somma(cost_i * x_i) <= budget_max
    """
    prob = pu.LpProblem("MktCampaignOptimization", pu.LpMaximize)
    n = len(campaigns)

    # Variabili x_i >= 0, Intere
    x = [pu.LpVariable(f"x_{i}", lowBound=0, cat="Integer") for i in range(n)]

    # Funzione Obiettivo: max sum( net_profit_i * x_i )
    profit_expr = []
    for i, camp in enumerate(campaigns):
        profit_expr.append(camp["net_profit"] * x[i])
    prob += pu.lpSum(profit_expr), "Total_Profit"

    # Vincolo (1): Somma lead = total_leads
    prob += pu.lpSum(x) == total_leads, "Totale_lead"

    # Vincolo (2): Somma lead 'corpo' >= corpo_percent * total_leads
    corpo_indices = [i for i, c in enumerate(campaigns) if c["category"] == "corpo"]
    if corpo_indices:
        prob += pu.lpSum([x[i] for i in corpo_indices]) >= corpo_percent * total_leads, "Minimo_corpo"

    # Vincolo (3): Ogni campagna >= min_share * somma di quella categoria
    cat_dict = defaultdict(list)
    for i, camp in enumerate(campaigns):
        cat_dict[camp["category"]].append(i)
    for category, indices in cat_dict.items():
        if len(indices) > 1:
            sum_cat = pu.lpSum([x[j] for j in indices])
            for j in indices:
                prob += x[j] >= min_share * sum_cat, f"MinShare_{category}_{j}"

    # Vincolo (4): somma(cost_i * x_i) <= budget_max
    cost_expr = [c["cost"] * x[i] for i, c in enumerate(campaigns)]
    prob += pu.lpSum(cost_expr) <= budget_max, "BudgetMax"

    # Risolve
    prob.solve(pu.PULP_CBC_CMD(msg=0))
    status = pu.LpStatus[prob.status]
    if status == "Optimal":
        x_values = [pu.value(var) for var in x]
        profit = pu.value(prob.objective)
        return status, x_values, profit
    else:
        return status, None, None

def compute_solution_df(campaigns, x_values):
    """
    Dato l'array x_values e la lista campaigns, 
    crea un DataFrame con leads, costi, ricavi, margine e una riga TOTALE in fondo.
    """
    results = []
    cost_sum = 0
    revenue_sum = 0
    profit_sum = 0
    lead_sum = 0

    for i, camp in enumerate(campaigns):
        leads = int(round(x_values[i] or 0))
        cost_t = camp["cost"] * leads
        revenue_t = camp["revenue"] * leads
        margin_t = camp["net_profit"] * leads

        cost_sum += cost_t
        revenue_sum += revenue_t
        profit_sum += margin_t
        lead_sum += leads

        results.append({
            "Campagna": camp["name"],
            "Categoria": camp["category"],
            "Leads": leads,
            "Costo Tot": int(round(cost_t)),
            "Ricavo Tot": int(round(revenue_t)),
            "Margine": int(round(margin_t))
        })

    df = pd.DataFrame(results)
    # Riga TOTALE
    df.loc["TOTALE"] = [
        "",  # Campagna
        "",  # Categoria
        lead_sum,
        int(round(cost_sum)),
        int(round(revenue_sum)),
        int(round(profit_sum))
    ]

    return df, cost_sum, revenue_sum, profit_sum, lead_sum

def main():
    st.title("Analisi di Scenario (Budget Limitato vs. Senza Vincolo)")
    st.write("""
    Questa app risolve due scenari:
    1. Scenario A: con un certo `budget_max_A`.
    2. Scenario B: con un budget molto alto (praticamente illimitato).

    Poi confronta i risultati, giustificando la spesa extra 
    e mostrando l'effetto sul margine e sul numero di lead prodotti.
    """)

    # Caricamento dati
    mode = st.radio(
        "Come vuoi inserire i dati? Carica un CSV come [questo file di esempio](https://drive.google.com/file/d/1vfp_gd6ivHsVpxffn_seAB11qCy9m0bP/view?usp=sharing)",
        ["Carica CSV", "Inserimento manuale"]
    )
    campaigns = []
    if mode == "Carica CSV":
        uploaded_file = st.file_uploader("Seleziona il tuo CSV", type=["csv"])
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.write("**Anteprima CSV:**")
            st.dataframe(df.head())

            df.columns = [c.lower().strip() for c in df.columns]
            required = ["nome campagna", "categoria campagna", "costo per lead", "ricavo per lead"]
            if all(r in df.columns for r in required):
                for _, row in df.iterrows():
                    name = row["nome campagna"]
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
                st.error(f"Mancano colonne: {required}. Controlla il CSV.")
    else:
        n = st.number_input("Numero di campagne (2..10):", min_value=2, max_value=10, value=2)
        for i in range(n):
            st.markdown(f"**Campagna #{i+1}**")
            name = st.text_input(f"Nome campagna #{i+1}", key=f"name_{i}")
            category = st.selectbox(f"Categoria #{i+1}", ["laser", "corpo"], key=f"cat_{i}")
            cost_ = st.number_input(f"Costo per lead #{i+1}", min_value=0.0, value=10.0, step=1.0, key=f"cost_{i}")
            revenue_ = st.number_input(f"Ricavo per lead #{i+1}", min_value=0.0, value=30.0, step=1.0, key=f"rev_{i}")
            net_profit = revenue_ - cost_

            campaigns.append({
                "name": name if name else f"Camp_{i+1}",
                "category": category,
                "cost": cost_,
                "revenue": revenue_,
                "net_profit": net_profit
            })

    st.write("---")
    if len(campaigns) == 0:
        st.info("Carica il CSV o inserisci i dati manualmente.")
        return

    st.subheader("Parametri di Ottimizzazione")
    total_leads = st.number_input("Totale dei lead da produrre (Scenario A e B):", min_value=1, value=10000, step=1)
    corpo_percent = st.slider("Percentuale minima di lead 'corpo':", 0.0, 1.0, 0.33, 0.01)
    min_share = st.slider("Percentuale minima su OGNI campagna nella stessa categoria:", 0.0, 1.0, 0.2, 0.01)

    st.write("**Scenario A**: Budget limitato")
    budget_max_A = st.number_input("Budget massimo per Scenario A:", min_value=0.0, value=50000.0, step=100.0)

    # Scenario B: budget "infinito" (ad es. 1e9)
    budget_max_B = 1e9

    if st.button("Esegui Analisi di Scenario"):
        # 1) Risolvi Scenario A
        statusA, xA, profitA = solve_mip(campaigns, total_leads, corpo_percent, min_share, budget_max_A)
        if statusA != "Optimal":
            st.error(f"Scenario A non ottimale o infeasible. Status: {statusA}")
            return
        dfA, costA, revenueA, marginA, leadsA = compute_solution_df(campaigns, xA)

        # 2) Risolvi Scenario B
        statusB, xB, profitB = solve_mip(campaigns, total_leads, corpo_percent, min_share, budget_max_B)
        if statusB != "Optimal":
            st.error(f"Scenario B non ottimale o infeasible. Status: {statusB}")
            return
        dfB, costB, revenueB, marginB, leadsB = compute_solution_df(campaigns, xB)

        # Mostra risultati SCENARIO A
        st.write("## Risultati Scenario A (Budget limitato)")
        st.table(dfA)
        st.write(f"Costo totale: {int(round(costA))} (<= {int(round(budget_max_A))} €)")
        st.write(f"Ricavo totale: {int(round(revenueA))}")
        st.write(f"Profitto totale: {int(round(marginA))}")
        st.write(f"Lead totali: {int(leadsA)}")

        # Mostra risultati SCENARIO B
        st.write("## Risultati Scenario B (Budget illimitato)")
        st.table(dfB)
        st.write(f"Costo totale: {int(round(costB))} (Budget altissimo, non vincolato)")
        st.write(f"Ricavo totale: {int(round(revenueB))}")
        st.write(f"Profitto totale: {int(round(marginB))}")
        st.write(f"Lead totali: {int(leadsB)}")

        st.write("---")
        st.subheader("Analisi di Scenario: Confronto A vs B")

        # Differenze
        extra_cost = costB - costA
        extra_margin = marginB - marginA
        extra_leads = leadsB - leadsA
        # margine ulteriore NETTO (extra_margin - extra_cost), se vuoi interpretarlo così
        extra_net = extra_margin - extra_cost

        st.write(f"**Spesa Scenario A**: {int(round(costA))} €, "
                 f"Scenario B: {int(round(costB))} €  "
                 f"(Differenza = {int(round(extra_cost))} €).")

        st.write(f"**Margine Scenario A**: {int(round(marginA))} €, "
                 f"Scenario B: {int(round(marginB))} €  "
                 f"(Differenza = {int(round(extra_margin))} €).")

        st.write(f"**Lead Scenario A**: {int(leadsA)}  "
                 f"Scenario B: {int(leadsB)}  "
                 f"(Differenza = {int(extra_leads)}).")

        st.write("___")

        if extra_cost > 0:
            st.markdown(f"""
            - Spendendo **{int(round(extra_cost))} €** in più, 
            - Ottenete **{int(round(extra_margin))} €** di margine aggiuntivo 
              (rispetto a Scenario A).
            - Se consideriamo il **guadagno netto** = differenza margine - differenza costi,
              otteniamo **{int(round(extra_net))} €** di profitto in più **al netto** 
              della spesa extra.
            - Avete prodotto **{int(extra_leads)}** lead in più.
            """)
        else:
            st.markdown("""
            In questo caso, lo scenario B non spende più di A 
            (o addirittura spende uguale/meno), 
            per cui non c’è un vero “sforamento” di budget.
            """)

if __name__ == "__main__":
    main()
