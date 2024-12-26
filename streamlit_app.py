import streamlit as st
import pandas as pd
import pulp as pu
from collections import defaultdict
import io

# Per l'esportazione in PDF
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors

def solve_with_pulp_integer(campaigns, total_leads, corpo_percent, min_share, budget_max):
    """
    Risolve il problema con PuLP, forzando x_i (lead per campagna i) a essere interi.
    
    Parametri:
    - campaigns: lista di dict {name, category, cost, revenue, net_profit}
    - total_leads: numero totale di lead da generare (int)
    - corpo_percent: percentuale minima di lead 'corpo' (0..1)
    - min_share: percentuale minima per OGNI campagna in una categoria (0..1)
    - budget_max: budget massimo disponibile (>= 0)

    Variabili: x_i >= 0, cat='Integer'

    Vincoli:
      1) somma(x_i) = total_leads
      2) somma(x_i in 'corpo') >= corpo_percent * total_leads
      3) per ogni categoria, OGNI campagna j ha x_j >= min_share * (somma x in cat)
      4) somma(cost_i * x_i) <= budget_max

    Obiettivo: max sum(net_profit_i * x_i)
    """

    prob = pu.LpProblem("MarketingCampaignOptimizationInteger", pu.LpMaximize)
    n = len(campaigns)

    # 1) Variabili di decisione (x_i, intere)
    x = [pu.LpVariable(f"x_{i}", lowBound=0, cat="Integer") for i in range(n)]

    # 2) Funzione obiettivo (profitto totale)
    profit_expr = [camp["net_profit"] * x[i] for i, camp in enumerate(campaigns)]
    prob += pu.lpSum(profit_expr), "Total_Profit"

    # 3) Vincoli
    # (a) Somma dei lead = total_leads
    prob += pu.lpSum(x) == total_leads, "Totale_lead"

    # (b) Somma dei lead 'corpo' >= corpo_percent * total_leads
    corpo_indices = [i for i, camp in enumerate(campaigns) if camp["category"] == "corpo"]
    if corpo_indices:
        prob += pu.lpSum([x[i] for i in corpo_indices]) >= corpo_percent * total_leads, "Minimo_corpo"

    # (c) OGNI campagna in una categoria ha almeno min_share
    cat_dict = defaultdict(list)
    for i, camp in enumerate(campaigns):
        cat_dict[camp["category"]].append(i)
    for category, indices in cat_dict.items():
        if len(indices) > 1:  # se c'è >= 2 campagne
            sum_in_cat = pu.lpSum(x[j] for j in indices)
            for j in indices:
                prob += x[j] >= min_share * sum_in_cat, f"MinShare_{category}_{j}"

    # (d) Vincolo di budget: somma(cost_i * x_i) <= budget_max
    cost_expr = [camp["cost"] * x[i] for i, camp in enumerate(campaigns)]
    prob += pu.lpSum(cost_expr) <= budget_max, "BudgetMax"

    # 4) Risoluzione (solver CBC, silenzioso)
    prob.solve(pu.PULP_CBC_CMD(msg=0))
    status = pu.LpStatus[prob.status]

    if status == "Optimal":
        x_values = [pu.value(var) for var in x]
        total_profit = pu.value(prob.objective)
        return status, x_values, total_profit
    else:
        return status, None, None

def main():
    st.title("Ottimizzatore di Campagne (OGNI campagna con min. percentuale) + Export XLSX/PDF")
    st.write("""
    - I lead sono **interi** (cat="Integer").
    - Vincolo su 'corpo_percent' e su 'budget massimo'.
    - **Nuovo**: OGNI campagna in ogni categoria deve avere **almeno** `min_share` della somma dei lead di quella categoria.
    - Attenzione a non superare 1.0 quando moltiplichi min_share per il numero di campagne in una categoria!
    """)

    st.subheader("1) Caricamento dei Dati")

    mode = st.radio(
        "Come vuoi inserire i dati delle tue campagne? Carica un CSV come [questo file di esempio](https://drive.google.com/file/d/1vfp_gd6ivHsVpxffn_seAB11qCy9m0bP/view?usp=sharing)",
        ["Carica CSV", "Inserimento manuale"]
    )
    campaigns = []

    if mode == "Carica CSV":
        uploaded_file = st.file_uploader("Seleziona il tuo CSV", type=["csv"])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            # Mostra un'anteprima
            st.write("**Anteprima del CSV caricato:**")
            st.dataframe(df.head())

            # Normalizziamo i nomi delle colonne
            df.columns = [c.lower().strip() for c in df.columns]
            required_cols = ["nome campagna", "categoria campagna", "costo per lead", "ricavo per lead"]
            if all(col in df.columns for col in required_cols):
                for _, row in df.iterrows():
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

        total_leads = st.number_input(
            "Totale dei lead da produrre (intero):", 
            min_value=1, value=10000, step=1
        )
        corpo_percent = st.slider(
            "Percentuale minima di lead 'corpo' (0% = 0.0, 100% = 1.0):", 
            min_value=0.0, max_value=1.0, value=0.33, step=0.01
        )
        min_share = st.slider(
            "Percentuale minima di lead per OGNI campagna (nella stessa categoria)",
            min_value=0.0, max_value=1.0, value=0.2, step=0.01
        )
        budget_max = st.number_input(
            "Budget massimo (EUR) che non vuoi superare:", 
            min_value=0.0, 
            value=50000.0, 
            step=100.0
        )

        if st.button("Esegui Ottimizzazione"):
            # Richiama la funzione di ottimizzazione con PuLP
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

            # Prepara la tabella di risultati
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

            corpo_leads_used = sum(r["Leads"] for r in results if r["Categoria"] == "corpo")

            st.write("**Riepilogo**")
            st.write(f"Totale lead: {int(total_leads)}")
            st.write(f"Lead 'corpo': {corpo_leads_used} (≥ {int(round(corpo_percent*100))}% del totale)")
            st.write(f"Lead 'laser': {int(total_leads - corpo_leads_used)}")
            st.write(f"Costo totale: {int(round(cost_total_sum))} (non supera {int(budget_max)}€)")
            st.write(f"Ricavo totale: {int(round(revenue_total_sum))}")
            st.write(f"Profitto totale: {int(round(profit_total_sum))}")

            # ==============================
            # EXPORT in EXCEL e PDF
            # ==============================
            df_result = pd.DataFrame(results)

            # ----- Esportazione in Excel -----
            import xlsxwriter
            import io

            xlsx_buffer = io.BytesIO()
            with pd.ExcelWriter(xlsx_buffer, engine="xlsxwriter") as writer:
                df_result.to_excel(writer, index=False, sheet_name="Risultati")

            st.download_button(
                label="Esporta in Excel (XLSX)",
                data=xlsx_buffer.getvalue(),
                file_name="risultati.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

            # ----- Esportazione in PDF -----
            pdf_buffer = io.BytesIO()
            doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)

            # Prepariamo i dati per la tabella PDF
            table_data = [["Campagna", "Categoria", "Leads", "Costo Tot", "Ricavo Tot", "Margine"]]
            for row in results:
                table_data.append([
                    row["Campagna"],
                    row["Categoria"],
                    str(row["Leads"]),
                    str(row["Costo Tot"]),
                    str(row["Ricavo Tot"]),
                    str(row["Margine"])
                ])

            pdf_table = Table(table_data)
            style = TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.gray),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
                ('GRID', (0,0), (-1,-1), 0.5, colors.black),
                ('ALIGN', (2,1), (-1,-1), 'RIGHT')
            ])
            pdf_table.setStyle(style)

            doc.build([pdf_table])
            pdf_buffer.seek(0)

            st.download_button(
                label="Esporta in PDF",
                data=pdf_buffer,
                file_name="risultati.pdf",
                mime="application/pdf"
            )

        else:
            st.info("Imposta i parametri e clicca su 'Esegui Ottimizzazione'.")
    else:
        st.info("Carica un CSV valido o inserisci le campagne manualmente.")

if __name__ == "__main__":
    main()
