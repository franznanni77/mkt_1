

import streamlit as st
import pandas as pd
import math
import scikit-learn as scikit


def allocate_greedy(category_camps, leads_cat):
    """
    Distribuisce 'leads_cat' tra le campagne in 'category_camps' in modo semplificato:
      - Se c'è 1 sola campagna, prende tutti i lead.
      - Se >= 2, assegna il 20% alla campagna col profitto unitario min,
        e l'80% a quella col profitto max. Le eventuali intermedie ricevono 0.
    Ritorna una lista di tuple (camp_name, leads_assigned, cost, revenue, profit).
    """
    if leads_cat <= 0:
        # Tutte a zero
        return [(c["name"], 0, 0, 0, 0) for c in category_camps]

    if len(category_camps) == 1:
        c = category_camps[0]
        assigned = leads_cat
        cost_tot = assigned * c["cost"]
        rev_tot = assigned * c["revenue"]
        prof_tot = assigned * c["net_profit"]
        return [(c["name"], assigned, cost_tot, rev_tot, prof_tot)]
    
    # Se >=2 campagne
    c_min = min(category_camps, key=lambda x: x["net_profit"])
    c_max = max(category_camps, key=lambda x: x["net_profit"])

    min_leads = int(round(0.2 * leads_cat))
    max_leads = leads_cat - min_leads

    cost_min = min_leads * c_min["cost"]
    rev_min = min_leads * c_min["revenue"]
    prof_min = min_leads * c_min["net_profit"]

    cost_max = max_leads * c_max["cost"]
    rev_max = max_leads * c_max["revenue"]
    prof_max = max_leads * c_max["net_profit"]

    alloc_list = [
        (c_min["name"], min_leads, cost_min, rev_min, prof_min),
        (c_max["name"], max_leads, cost_max, rev_max, prof_max)
    ]

    # Per eventuali altre campagne in mezzo (non min, non max), assegniamo 0.
    for c in category_camps:
        if c["name"] not in [c_min["name"], c_max["name"]]:
            alloc_list.append((c["name"], 0, 0, 0, 0))

    return alloc_list

def compute_best_distribution(campaigns, total_leads, corpo_percent):
    """
    Ricerca esaustiva su quanti lead assegnare a 'corpo' (da min_corpo_leads a total_leads).
    Restituisce la soluzione con profitto massimo.
    """
    # Suddividiamo le campagne per categoria
    laser_camps = [c for c in campaigns if c["category"] == "laser"]
    corpo_camps = [c for c in campaigns if c["category"] == "corpo"]

    min_corpo_leads = int(math.ceil(corpo_percent * total_leads))

    best_solution = None
    best_profit = -1e9

    for corpo_leads_assigned in range(min_corpo_leads, total_leads + 1):
        laser_leads_assigned = total_leads - corpo_leads_assigned

        # Alloca in modo "greedy"
        corpo_alloc = allocate_greedy(corpo_camps, corpo_leads_assigned)
        laser_alloc = allocate_greedy(laser_camps, laser_leads_assigned)

        # Profitto totale
        total_profit = sum(item[4] for item in corpo_alloc) + sum(item[4] for item in laser_alloc)

        if total_profit > best_profit:
            best_profit = total_profit
            best_solution = (corpo_alloc, laser_alloc, corpo_leads_assigned, laser_leads_assigned)

    return best_solution, best_profit

def main():
    st.title("Ottimizzatore di Campagne (Senza Solver)")
    st.write("""
    Questa applicazione dimostra una strategia di ottimizzazione 'fatta in casa':
    - Suddivide i lead tra 'corpo' e 'laser' in tutti i valori possibili compatibili con la percentuale minima di 'corpo'.
    - All'interno di ogni categoria assegna il 20% alla campagna col profitto min, e l'80% a quella col profitto max (se >=2 campagne).
    - **Limitazioni**: se in una categoria ci sono 3 o più campagne, la soluzione potrebbe non essere davvero ottimale.
    """)

    st.subheader("1) Caricamento dei Dati")

    # Scelta modalità: CSV o input manuale
    mode = st.radio("Come vuoi inserire i dati delle campagne?", ["Carica CSV", "Inserimento manuale"])

    campaigns = []

    if mode == "Carica CSV":
        st.write("Carica un file CSV con queste colonne: **nome campagna, categoria campagna, costo per lead, ricavo per lead**.")
        uploaded_file = st.file_uploader("Seleziona il tuo CSV", type=["csv"])
        
        if uploaded_file is not None:
            # Leggiamo il CSV con pandas
            df = pd.read_csv(uploaded_file)
            # Ci aspettiamo le colonne: 
            # "nome campagna", "categoria campagna", "Costo per lead", "ricavo per lead"
            # Normalizziamo i nomi colonna (minuscoli)
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

        if st.button("Esegui Ottimizzazione"):
            best_solution, best_profit = compute_best_distribution(campaigns, int(total_leads), corpo_percent)
            if best_solution is None:
                st.error("Nessuna soluzione trovata. (Imprevisto!)")
                return

            corpo_alloc, laser_alloc, corpo_leads_used, laser_leads_used = best_solution

            # Creiamo una tabella da mostrare
            results = []
            # corpo_alloc e laser_alloc contengono tuple: (camp_name, leads, cost, rev, profit)
            all_alloc = corpo_alloc + laser_alloc
            for (name, leads, cost_t, rev_t, prof_t) in all_alloc:
                results.append({
                    "Campagna": name,
                    "Leads": int(round(leads)),
                    "Costo Tot": int(round(cost_t)),
                    "Ricavo Tot": int(round(rev_t)),
                    "Margine": int(round(prof_t))
                })
            
            st.write("**Assegnazione Campagne**")
            st.table(results)

            tot_cost = sum(r["Costo Tot"] for r in results)
            tot_revenue = sum(r["Ricavo Tot"] for r in results)
            tot_profit = sum(r["Margine"] for r in results)

            st.write("**Riepilogo**")
            st.write(f"Totale lead: {int(total_leads)}")
            st.write(f"Lead 'corpo': {corpo_leads_used} (≥ {int(round(corpo_percent*100))}% del totale)")
            st.write(f"Lead 'laser': {laser_leads_used}")
            st.write(f"Costo totale: {tot_cost}")
            st.write(f"Ricavo totale: {tot_revenue}")
            st.write(f"Profitto totale: {tot_profit}")
        else:
            st.info("Imposta i parametri e poi clicca su 'Esegui Ottimizzazione'.")

    else:
        st.info("Carica un CSV valido oppure inserisci le campagne manualmente.")

if __name__ == "__main__":
    main()
