import streamlit as st
import pandas as pd
import pulp as pu
from collections import defaultdict
import io

# Per PDF
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

    Ritorna: (status, x_values, profit)
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

    # Vincolo (3): Ogni campagna >= min_share * (somma x in cat)
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

    # Risolvi
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
    Dato x_values e campaigns, crea un DataFrame con:
      Campagna, Categoria, Leads, Costo Tot, Ricavo Tot, Margine
    e aggiunge in fondo una riga TOTALE.
    Ritorna (df, cost_sum, revenue_sum, profit_sum, lead_sum)
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
    # Aggiungiamo la riga TOTALE
    df.loc["TOTALE"] = [
        "",  # Campagna
        "",  # Categoria
        lead_sum,
        int(round(cost_sum)),
        int(round(revenue_sum)),
        int(round(profit_sum))
    ]
    return df, cost_sum, revenue_sum, profit_sum, lead_sum

def export_to_excel(df, scenario_name="Scenario"):
    """
    Esporta il DataFrame df in un buffer Excel (BytesIO), 
    usando scenario_name nel file_name.
    """
    import xlsxwriter

    xlsx_buffer = io.BytesIO()
    with pd.ExcelWriter(xlsx_buffer, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=True, sheet_name="Risultati")
    file_name = f"risultati_{scenario_name}.xlsx"
    return xlsx_buffer.getvalue(), file_name

def export_to_pdf(df, scenario_name="Scenario"):
    """
    Esporta il DataFrame df in un PDF (BytesIO), 
    usando scenario_name nel file_name.
    """
    pdf_buffer = io.BytesIO()
    doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)

    # Prepara i dati per la tabella
    table_data = [list(df.columns)]  # header
    for idx, row in df.iterrows():
        table_data.append([
            str(row["Campagna"]),
            str(row["Categoria"]),
            str(row["Leads"]),
            str(row["Costo Tot"]),
            str(row["Ricavo Tot"]),
            str(row["Margine"]),
        ])

    table = Table(table_data)
    style = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.gray),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (2, 1), (-1, -1), 'RIGHT'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
    ])
    table.setStyle(style)

    doc.build([table])
    pdf_buffer.seek(0)

    file_name = f"risultati_{scenario_name}.pdf"
    return pdf_buffer.getvalue(), file_name

def main():
    st.title("Analisi di Scenario (Budget Limitato vs. Senza Vincolo)")
    st.write("""
    Questa app risolve due scenari:
    1. **Scenario A**: con un certo `budget_max_A`.
    2. **Scenario B**: con un budget molto alto (di fatto illimitato).

    Poi confronta i risultati, giustificando la spesa extra 
    e mostrando l'effetto sul margine e sul numero di lead prodotti.
    Ogni scenario può anche essere esportato in PDF o Excel.
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
                    name = str(row["nome campagna"]).strip()
                    category = str(row["categoria campagna"]).lower().strip()
                    cost_ = float(row["costo per lead"])
                    revenue_ = float(row["ricavo per lead"])
                    net_profit = revenue_ - cost_

                    campaigns.append({
                        "name": name,
                        "category": category,
                        "cost": cost_,
                        "revenue": revenue_,
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
    total_leads = st.number_input("Totale dei lead da produrre:", min_value=1, value=10000, step=1)
    corpo_percent = st.slider("Percentuale minima di lead 'corpo':", 0.0, 1.0, 0.33, 0.01)
    min_share = st.slider("Percentuale minima su OGNI campagna nella stessa categoria:", 0.0, 1.0, 0.2, 0.01)

    st.write("**Scenario A**: Budget limitato")
    budget_max_A = st.number_input("Budget massimo per Scenario A:", min_value=0.0, value=50000.0, step=100.0)

    st.write("**Scenario B**: Budget illimitato (non modificabile)", 1e9)

    if st.button("Esegui Analisi di Scenario"):
        # Risolvi Scenario A (budget limitato)
        statusA, xA, profitA = solve_mip(campaigns, total_leads, corpo_percent, min_share, budget_max_A)
        if statusA != "Optimal":
            st.error(f"Scenario A non ottimale o infeasible. Status: {statusA}")
            return

        dfA, costA, revenueA, marginA, leadsA = compute_solution_df(campaigns, xA)

        # Risolvi Scenario B (budget = 1e9)
        statusB, xB, profitB = solve_mip(campaigns, total_leads, corpo_percent, min_share, 1e9)
        if statusB != "Optimal":
            st.error(f"Scenario B non ottimale o infeasible. Status: {statusB}")
            return

        dfB, costB, revenueB, marginB, leadsB = compute_solution_df(campaigns, xB)

        # SCENARIO A - RISULTATI + ESPORTAZIONE
        st.write("## Risultati Scenario A (Budget limitato)")
        st.table(dfA)
        st.write(f"Costo totale: {int(round(costA))} (<= {int(round(budget_max_A))} €)")
        st.write(f"Ricavo totale: {int(round(revenueA))}")
        st.write(f"Profitto totale: {int(round(marginA))}")
        st.write(f"Lead totali: {int(leadsA)}")

        # Download Scenario A
        excelA, fnameA_xlsx = export_to_excel(dfA, scenario_name="ScenarioA")
        pdfA, fnameA_pdf = export_to_pdf(dfA, scenario_name="ScenarioA")

        st.download_button(
            label="Esporta Scenario A in Excel",
            data=excelA,
            file_name=fnameA_xlsx,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        st.download_button(
            label="Esporta Scenario A in PDF",
            data=pdfA,
            file_name=fnameA_pdf,
            mime="application/pdf"
        )

        st.write("---")

        # SCENARIO B - RISULTATI + ESPORTAZIONE
        st.write("## Risultati Scenario B (Budget illimitato)")
        st.table(dfB)
        st.write(f"Costo totale: {int(round(costB))} (budget altissimo, non vincolato)")
        st.write(f"Ricavo totale: {int(round(revenueB))}")
        st.write(f"Profitto totale: {int(round(marginB))}")
        st.write(f"Lead totali: {int(leadsB)}")

        # Download Scenario B
        excelB, fnameB_xlsx = export_to_excel(dfB, scenario_name="ScenarioB")
        pdfB, fnameB_pdf = export_to_pdf(dfB, scenario_name="ScenarioB")

        st.download_button(
            label="Esporta Scenario B in Excel",
            data=excelB,
            file_name=fnameB_xlsx,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        st.download_button(
            label="Esporta Scenario B in PDF",
            data=pdfB,
            file_name=fnameB_pdf,
            mime="application/pdf"
        )

        st.write("---")

        # ANALISI CONFRONTO
        st.subheader("Analisi di Scenario: Confronto A vs B")

        extra_cost = costB - costA
        extra_margin = marginB - marginA
        extra_leads = leadsB - leadsA
        extra_net = extra_margin - extra_cost  # se vuoi interpretare il guadagno netto

        st.write(f"**Scenario A** spende {int(round(costA))} € e ottiene un margine di {int(round(marginA))} €, producendo {int(leadsA)} lead.")
        st.write(f"**Scenario B** spende {int(round(costB))} € e ottiene un margine di {int(round(marginB))} €, producendo {int(leadsB)} lead.")

        if extra_cost > 0:
            st.markdown(f"""
            - Spendendo **{int(round(extra_cost))} €** in più (rispetto a Scenario A),
            - ottieni **{int(round(extra_margin))} €** di margine aggiuntivo
            - produci **{int(extra_leads)}** lead in più
            - il **guadagno netto** extra (margine extra - spesa extra) è **{int(round(extra_net))} €**.
            """)
        else:
            st.write("Lo Scenario B non spende più di A (o addirittura meno), quindi non c'è sforamento di budget.")

if __name__ == "__main__":
    main()
