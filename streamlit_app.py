import streamlit as st
import pandas as pd
import math
import pulp as pu  # PuLP
from collections import defaultdict

def solve_with_pulp(campaigns, total_leads, corpo_percent):
    """
    Risolve il problema con PuLP:
      - x_i = numero di lead per la campagna i (Continuous o Integer).
      - Vincoli:
        1) somma(x_i) = total_leads
        2) somma(x_i) su 'corpo' >= corpo_percent * total_leads
        3) per ogni categoria con >=2 campagne, la campagna meno profittevole
           ottiene >= 20% dei lead di quella categoria
      - Obiettivo: max somma( net_profit_i * x_i )
    Restituisce (status, x_i_dict, profit_total).
    """

    # 1. Creazione modello
    prob = pu.LpProblem("MarketingCampaignOptimization", pu.LpMaximize)

    n = len(campaigns)

    # 2. Definizione variabili di decisione
    # Se vuoi forzare lead interi, usa cat="Integer"
    x = [pu.LpVariable(f"x_{i}", lowBound=0, cat="Continuous") for i in range(n)]

    # 3. Funzione obiettivo
    profit_expr = []
    for i, camp in enumerate(campaigns):
        profit_expr.append(camp["net_profit"] * x[i])
    prob += pu.lpSum(profit_expr), "Total_Profit"

    # 4. Vincoli

    # (a) Somma dei lead = total_leads
    prob += pu.lpSum(x) == total_leads, "Totale_lead"

    # (b) Somma dei lead 'corpo' >= corpo_percent * total_leads
    corpo_indices = [i for i, camp in enumerate(campaigns) if camp["category"] == "corpo"]
    if corpo_indices:
        prob += pu.lpSum([x[i] for i in corpo_indices]) >= corpo_percent * total_leads, "Minimo_corpo"

    # (c) Vincolo 20% alla campagna meno profittevole (per ogni categoria con >=2 campagne)
    cat_dict = defaultdict(list)
    for i, camp in enumerate(campaigns):
        cat_dict[camp["category"]].append(i)

    for category, indices in cat_dict.items():
        if len(indices) > 1:
            # Trova l'indice della campagna con net_profit minimo
            min_i = min(indices, key=lambda idx: campaigns[idx]["net_profit"])
            sum_in_cat = pu.lpSum([x[idx] for idx in indices])
            # x_min >= 0.2 * somma_x_in_cat
            prob += x[min_i] >= 0.2 * sum_in_cat, f"MinProfit_20pct_{category}"

    # 5. Risoluzione
    prob.solve(pu.PULP_CBC_CMD(msg=0))
    status = pu.LpStatus[prob.status]

    # 6. Lettura risultati
    if status == "Optimal":
        x_values = [pu.value(var) for var in x]
        total_profit = pu.value(prob.objective)
        return status, x_values, total_profit
    else:
        return status, None, None

def main():
    st.title("Ottimizzatore di Campagne con PuLP")
    st.write("""
    In questa versione, usiamo **PuLP** per creare e risolvere un modello di programmazione lineare:
    - Ogni campagna ha una variabile x_i = lead assegnati.
    - Somma(x_i) = Totale lead.
    - Almeno corpo_percent * Totale lead ai 'corpo'.
    - In ogni categoria con >=2 campagne, quella meno profittevole riceve >= 20% dei lead di quella categoria.
    - Obiettivo: massimizzare la somma dei margini (ricavo-costo).
    """)

    st.subheader("1) Caricamento dei Dati")

    mode = st.radio("Come vuoi inserire i dati delle campagne?", ["Carica CSV", "Inserimento manuale"])
    campaigns = []

    if mode == "Carica CSV":
        st.write("Carica un file CSV con queste colonne: **nome campagna, categoria campagna, costo per lead, ricavo per lead**.")
        uploaded_file = st.file_uploader("Seleziona il tuo CSV", type=["csv"])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            # Normalizziamo le colonne
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
        # Inserimento manuale
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

        total_leads = st.number_input("Totale dei lead da produrre:", min_value=1, value=10000, step=1)
        corpo_percent = st.slider("Percentuale minima di lead 'corpo' (0% = 0.0, 100% = 1.0):", 
                                  min_value=0.0, max_value=1.0, value=0.33, step=0.01)
        
        # Se vuoi l'obbligo che x_i siano interi, aggiungi un toggle:
        use_integer = st.checkbox("Forza numero di lead interi (Mixed Integer Programming)")

        if st.button("Esegui Ottimizzazione"):

            # Se vogliamo integer, cambiamo la cat="Integer" direttamente nella funzione solve_with_pulp
            # Qui un esempio veloce di come farlo: creiamo un clone con "Integer"
            if use_integer:
                # Re-definiamo un piccolo override
                status, x_values, total_profit = solve_with_pulp_integer(campaigns, total_leads, corpo_percent)
            else:
                status, x_values, total_profit = solve_with_pulp(campaigns, total_leads, corpo_percent)

            if status != "Optimal":
                st.error(f"Soluzione non ottimale o infeasible. Stato: {status}")
                return

            # x_values ha i lead assegnati a ciascuna campagna
            results = []
            cost_total_sum = 0.0
            revenue_total_sum = 0.0
            profit_total_sum = 0.0

            for i, camp in enumerate(campaigns):
                leads_float = x_values[i] or 0.0
                # Se stai usando cat="Continuous", potresti avere decimali; se cat="Integer", avrai interi
                leads = int(round(leads_float))  # puoi anche lasciare float se preferisci

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
                    "Margine": int(round(margin_t)),
                })

            # Mostriamo la tabella
            st.write("**Assegnazione Campagne**")
            st.table(results)

            st.write("**Riepilogo**")
            st.write(f"Totale lead: {int(total_leads)}")
            # Calcoliamo quanti lead corpo sono stati assegnati
            corpo_leads_used = sum(
                r["Leads"] for r in results if r["Categoria"] == "corpo"
            )
            st.write(f"Lead 'corpo': {corpo_leads_used} (≥ {int(round(corpo_percent*100))}% del totale)")
            st.write(f"Lead 'laser': {int(total_leads - corpo_leads_used)}")

            st.write(f"Costo totale: {int(round(cost_total_sum))}")
            st.write(f"Ricavo totale: {int(round(revenue_total_sum))}")
            st.write(f"Profitto totale: {int(round(profit_total_sum))}")
        else:
            st.info("Imposta i parametri e clicca su 'Esegui Ottimizzazione'.")
    else:
        st.info("Carica un CSV valido oppure inserisci le campagne manualmente.")


def solve_with_pulp_integer(campaigns, total_leads, corpo_percent):
    """
    Variante per vincolare x_i a valori interi (Mixed Integer Programming).
    Quasi identica a solve_with_pulp ma con cat='Integer'.
    """
    from collections import defaultdict

    prob = pu.LpProblem("MarketingCampaignOptimizationInteger", pu.LpMaximize)
    n = len(campaigns)

    # Variabili x_i (intere)
    x = [pu.LpVariable(f"x_{i}", lowBound=0, cat="Integer") for i in range(n)]

    # Funzione obiettivo
    profit_expr = []
    for i, camp in enumerate(campaigns):
        profit_expr.append(camp["net_profit"] * x[i])
    prob += pu.lpSum(profit_expr), "Total_Profit"

    # Vincoli
    prob += pu.lpSum(x) == total_leads, "Totale_lead"

    corpo_indices = [i for i, camp in enumerate(campaigns) if camp["category"] == "corpo"]
    if corpo_indices:
        prob += pu.lpSum([x[i] for i in corpo_indices]) >= corpo_percent * total_leads, "Minimo_corpo"

    cat_dict = defaultdict(list)
    for i, camp in enumerate(campaigns):
        cat_dict[camp["category"]].append(i)

    for category, indices in cat_dict.items():
        if len(indices) > 1:
            min_i = min(indices, key=lambda idx: campaigns[idx]["net_profit"])
            sum_in_cat = pu.lpSum([x[idx] for idx in indices])
            prob += x[min_i] >= 0.2 * sum_in_cat, f"MinProfit_20pct_{category}"

    prob.solve(pu.PULP_CBC_CMD(msg=0))
    status = pu.LpStatus[prob.status]

    if status == "Optimal":
        x_values = [pu.value(var) for var in x]
        total_profit = pu.value(prob.objective)
        return status, x_values, total_profit
    else:
        return status, None, None


if __name__ == "__main__":
    main()
