import streamlit as st
from pulp import *
from collections import defaultdict

def main():
    st.title("Ottimizzatore di Campagne Marketing")
    st.write("""
    Questa applicazione utilizza la Programmazione Lineare (PuLP) per 
    distribuire i lead tra diverse campagne al fine di massimizzare il profitto.
    """)
    
    # 1. INPUT DELL'UTENTE
    # ---------------------------------------------------------------------------------
    st.subheader("Impostazioni iniziali")

    # Numero di campagne
    n = st.number_input(
        "Numero di campagne (minimo 2, massimo 10):",
        min_value=2, max_value=10, value=2, step=1
    )

    st.write("Inserisci i dati per ciascuna campagna:")

    # Prepariamo una lista che conterrà i dati di ogni campagna
    campaigns = []
    for i in range(n):
        st.markdown(f"**Campagna #{i+1}**")
        
        # Nome campagna
        name = st.text_input(f"Nome campagna #{i+1}", key=f"name_{i}")
        
        # Categoria: laser o corpo
        category = st.selectbox(
            f"Categoria campagna #{i+1}",
            options=["laser", "corpo"],
            key=f"cat_{i}"
        )
        
        # Costo per lead
        cost = st.number_input(
            f"Costo per lead (campagna #{i+1}):", 
            min_value=0.0, value=0.0, step=1.0, 
            format="%.2f",
            key=f"cost_{i}"
        )

        # Ricavo per lead
        revenue = st.number_input(
            f"Ricavo per lead (campagna #{i+1}):", 
            min_value=0.0, value=0.0, step=1.0,
            format="%.2f",
            key=f"revenue_{i}"
        )

        net_profit = revenue - cost

        # Salviamo i dati in un dizionario
        campaigns.append({
            "name": name if name else f"Campagna_{i+1}",
            "category": category,
            "cost": cost,
            "revenue": revenue,
            "net_profit": net_profit
        })

    # Numero totale di lead da generare
    total_leads = st.number_input(
        "Totale dei lead da produrre:", 
        min_value=1.0, value=10000.0, step=100.0
    )

    # Percentuale minima di lead 'corpo'
    corpo_percent = st.slider(
        "Percentuale minima di lead 'corpo' (0% = 0.0, 100% = 1.0):",
        min_value=0.0, max_value=1.0, value=0.33, step=0.01
    )

    st.markdown("---")
    st.write("""
    **Vincolo aggiuntivo**: 
    In ogni categoria che abbia almeno 2 campagne, 
    la **campagna meno profittevole** deve ricevere 
    **almeno il 20%** dei lead totali di quella categoria.
    """)

    # Pulsante per eseguire l'ottimizzazione
    if st.button("Esegui Ottimizzazione"):
        
        # 2. CREAZIONE DEL PROBLEMA DI LP
        # ---------------------------------------------------------------------------------
        prob = pulp.LpProblem("MarketingCampaignOptimization", pulp.LpMaximize)

        # Definiamo le variabili di decisione x_i
        x = {}
        for i, camp in enumerate(campaigns):
            x[i] = pulp.LpVariable(f"x_{i}", lowBound=0, cat="Continuous")

        # 3. FUNZIONE OBIETTIVO: massimizzare la somma dei profitti
        # ---------------------------------------------------------------------------------
        # Profitto totale = SUM( (ricavo - costo) * x_i )
        profit_expr = [camp["net_profit"] * x[i] for i, camp in enumerate(campaigns)]
        prob += pulp.lpSum(profit_expr), "Total_Profit"

        # 4. VINCOLI
        # ---------------------------------------------------------------------------------
        
        # (a) Somma dei lead = total_leads
        prob += pulp.lpSum([x[i] for i in x]) == total_leads, "Totale_lead"

        # (b) Somma dei lead 'corpo' >= corpo_percent * total_leads
        prob += pulp.lpSum(
            x[i] for i, camp in enumerate(campaigns)
            if camp["category"] == "corpo"
        ) >= corpo_percent * total_leads, "Minimo_corpo"

        # (c) Vincolo: nella stessa categoria, la campagna meno profittevole
        #     deve ricevere >= 20% dei lead di quella categoria.
        from collections import defaultdict
        cat_dict = defaultdict(list)
        for i, camp in enumerate(campaigns):
            cat_dict[camp["category"]].append(i)

        for category, indices in cat_dict.items():
            if len(indices) > 1:
                # Trova l'indice della campagna meno profittevole
                min_i = min(indices, key=lambda idx: campaigns[idx]["net_profit"])
                sum_in_cat = pulp.lpSum([x[idx] for idx in indices])
                x_min = x[min_i]
                # Vincolo: x_min >= 0.2 * sum_in_cat
                prob += x_min >= 0.2 * sum_in_cat, f"Vincolo_min_profit_cat_{category}"

        # 5. RISOLUZIONE DEL PROBLEMA
        # ---------------------------------------------------------------------------------
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        status = pulp.LpStatus[prob.status]

        # 6. OUTPUT RISULTATI
        # ---------------------------------------------------------------------------------
        st.subheader("Risultati dell'Ottimizzazione")
        st.write(f"Stato della soluzione: **{status}**")

        if status == "Optimal":
            # Valore della funzione obiettivo
            total_profit_float = pulp.value(prob.objective) or 0.0

            # Preparazione di variabili per sommare costi, ricavi, profitto
            cost_total_sum = 0.0
            revenue_total_sum = 0.0
            profit_total_sum = 0.0

            # Creiamo una struttura per mostrare i dati in tabella
            report_data = []

            # Calcoliamo i risultati campagna per campagna
            for i, camp in enumerate(campaigns):
                leads_float = pulp.value(x[i]) or 0.0
                leads = int(round(leads_float))  # converti a intero

                cost_total = camp["cost"] * leads
                revenue_total = camp["revenue"] * leads
                margin_total = camp["net_profit"] * leads

                cost_total_sum += cost_total
                revenue_total_sum += revenue_total
                profit_total_sum += margin_total

                # Calcolo % lead rispetto al totale
                if total_leads != 0:
                    lead_percentage = int(round((leads_float / total_leads) * 100))
                else:
                    lead_percentage = 0

                report_data.append({
                    "Campagna": camp['name'],
                    "Categoria": camp['category'],
                    "Leads": leads,
                    "Costo Totale": int(round(cost_total)),
                    "Ricavo Totale": int(round(revenue_total)),
                    "Margine": int(round(margin_total)),
                    "% Lead": f"{lead_percentage} %"
                })

            # Mostriamo i risultati in una tabella
            st.table(report_data)

            # Riepilogo finale
            total_leads_int = int(round(total_leads))
            st.write("**Numero di lead totali:**", total_leads_int)
            st.write("**Costo totale:**", int(round(cost_total_sum)))
            st.write("**Ricavo totale:**", int(round(revenue_total_sum)))
            st.write("**Profitto totale:**", int(round(profit_total_sum)))
        else:
            st.error("Non è stato possibile trovare una soluzione ottima con i vincoli impostati.")

if __name__ == "__main__":
    main()
