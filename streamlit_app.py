import streamlit as st
import pandas as pd
import pulp as pu
from collections import defaultdict
import io
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors

def solve_mip(
    campaigns, 
    total_leads, 
    corpo_percent, 
    min_share, 
    budget_max,
    weight_immediate,
):
    """
    Risolve il problema con PuLP, cat='Integer', 
    con i vincoli:
      1) somma(x_i) = total_leads
      2) somma(x_i in 'corpo') >= corpo_percent * total_leads
      3) per ogni categoria, x_j >= min_share * somma(x in cat)
      4) somma(cost_i * x_i) <= budget_max

    La profittabilità è calcolata come media pesata tra:
    - profitto immediato: (revenue - cost) * weight_immediate
    - profitto a 60gg: (revenue_60d - cost) * (1 - weight_immediate)
    """
    prob = pu.LpProblem("MktCampaignOptimization", pu.LpMaximize)
    n = len(campaigns)

    x = [pu.LpVariable(f"x_{i}", lowBound=0, cat="Integer") for i in range(n)]

    profit_expr = []
    for i, camp in enumerate(campaigns):
        immediate_profit = (camp["revenue"] - camp["cost"]) * weight_immediate
        profit_60d = (camp["revenue_60d"] - camp["cost"]) * (1 - weight_immediate)
        weighted_profit = immediate_profit + profit_60d
        profit_expr.append(weighted_profit * x[i])
    prob += pu.lpSum(profit_expr), "Total_Weighted_Profit"

    prob += pu.lpSum(x) == total_leads, "Totale_lead"

    corpo_indices = [i for i, c in enumerate(campaigns) if c["category"] == "corpo"]
    if corpo_indices:
        prob += pu.lpSum([x[i] for i in corpo_indices]) >= corpo_percent * total_leads, "Minimo_corpo"

    cat_dict = defaultdict(list)
    for i, camp in enumerate(campaigns):
        cat_dict[camp["category"]].append(i)
    for category, indices in cat_dict.items():
        if len(indices) > 1:
            sum_cat = pu.lpSum([x[j] for j in indices])
            for j in indices:
                prob += x[j] >= min_share * sum_cat, f"MinShare_{category}_{j}"

    cost_expr = [c["cost"] * x[i] for i, c in enumerate(campaigns)]
    prob += pu.lpSum(cost_expr) <= budget_max, "BudgetMax"

    prob.solve(pu.PULP_CBC_CMD(msg=0))
    status = pu.LpStatus[prob.status]
    if status == "Optimal":
        x_values = [pu.value(var) for var in x]
        profit = pu.value(prob.objective)
        return status, x_values, profit
    else:
        return status, None, None

def compute_solution_df(campaigns, x_values, weight_immediate):
    """
    Crea DataFrame con risultati e calcola totali
    """
    results = []
    cost_sum = 0
    revenue_sum = 0
    revenue_60d_sum = 0
    margin_immediate_sum = 0
    margin_60d_sum = 0
    weighted_margin_sum = 0
    lead_sum = 0

    for i, camp in enumerate(campaigns):
        leads = int(round(x_values[i] or 0))
        cost_t = camp["cost"] * leads
        revenue_t = camp["revenue"] * leads
        revenue_60d_t = camp["revenue_60d"] * leads
        
        margin_immediate_t = (camp["revenue"] - camp["cost"]) * leads
        margin_60d_t = (camp["revenue_60d"] - camp["cost"]) * leads
        weighted_margin_t = margin_immediate_t * weight_immediate + margin_60d_t * (1 - weight_immediate)

        cost_sum += cost_t
        revenue_sum += revenue_t
        revenue_60d_sum += revenue_60d_t
        margin_immediate_sum += margin_immediate_t
        margin_60d_sum += margin_60d_t
        weighted_margin_sum += weighted_margin_t
        lead_sum += leads

        results.append({
            "Campagna": camp["name"],
            "Categoria": camp["category"],
            "Leads": leads,
            "Costo Tot": int(round(cost_t)),
            "Ricavo Tot": int(round(revenue_t)),
            "Ricavo 60gg Tot": int(round(revenue_60d_t)),
            "Margine Immediato": int(round(margin_immediate_t)),
            "Margine 60gg": int(round(margin_60d_t)),
            "Margine Pesato": int(round(weighted_margin_t))
        })

    df = pd.DataFrame(results)
    df.loc["TOTALE"] = [
        "",
        "",
        lead_sum,
        int(round(cost_sum)),
        int(round(revenue_sum)),
        int(round(revenue_60d_sum)),
        int(round(margin_immediate_sum)),
        int(round(margin_60d_sum)),
        int(round(weighted_margin_sum))
    ]
    return df

def export_to_excel(df, scenario_name="Scenario"):
    xlsx_buffer = io.BytesIO()
    with pd.ExcelWriter(xlsx_buffer, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=True, sheet_name="Risultati")
    file_name = f"risultati_{scenario_name}.xlsx"
    return xlsx_buffer.getvalue(), file_name

def export_to_pdf(df, scenario_name="Scenario"):
    pdf_buffer = io.BytesIO()
    doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)

    table_data = [list(df.columns)]
    for idx, row in df.iterrows():
        table_data.append([
            str(row["Campagna"]),
            str(row["Categoria"]),
            str(row["Leads"]),
            str(row["Costo Tot"]),
            str(row["Ricavo Tot"]),
            str(row["Ricavo 60gg Tot"]),
            str(row["Margine Immediato"]),
            str(row["Margine 60gg"]),
            str(row["Margine Pesato"]),
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

    La profittabilità è calcolata come media pesata tra margine immediato e margine a 60 giorni.
    """)

    mode = st.radio(
        "Come vuoi inserire i dati?",
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
            required = ["nome campagna", "categoria campagna", "costo per lead", "ricavo per lead", "ricavo per lead a 60 giorni"]
            if all(r in df.columns for r in required):
                for _, row in df.iterrows():
                    name = str(row["nome campagna"]).strip()
                    category = str(row["categoria campagna"]).lower().strip()
                    cost_ = float(row["costo per lead"])
                    revenue_ = float(row["ricavo per lead"])
                    revenue_60d_ = float(row["ricavo per lead a 60 giorni"])

                    campaigns.append({
                        "name": name,
                        "category": category,
                        "cost": cost_,
                        "revenue": revenue_,
                        "revenue_60d": revenue_60d_
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
            revenue_60d_ = st.number_input(f"Ricavo per lead a 60gg #{i+1}", min_value=0.0, value=40.0, step=1.0, key=f"rev60_{i}")

            campaigns.append({
                "name": name if name else f"Camp_{i+1}",
                "category": category,
                "cost": cost_,
                "revenue": revenue_,
                "revenue_60d": revenue_60d_
            })

    st.write("---")
    if len(campaigns) == 0:
        st.info("Carica il CSV o inserisci i dati manualmente.")
        return

    st.subheader("Parametri di Ottimizzazione")
    weight_immediate = st.slider(
        "Peso profittabilità immediata (vs. 60gg):", 
        0.0, 1.0, 0.5, 0.01,
        help="1 = solo profitto immediato, 0 = solo profitto a 60gg"
    )
    total_leads = st.number_input("Totale dei lead da produrre:", min_value=1, value=10000, step=1)
    corpo_percent = st.slider("Percentuale minima di lead 'corpo':", 0.0, 1.0, 0.33, 0.01)
    min_share = st.slider("Percentuale minima su OGNI campagna nella stessa categoria:", 0.0, 1.0, 0.2, 0.01)

    st.write("**Scenario A**: Budget limitato")
    budget_max_A = st.number_input("Budget massimo per Scenario A:", min_value=0.0, value=50000.0, step=100.0)

    st.write("**Scenario B**: Budget illimitato (non modificabile)", 1e9)

    if st.button("Esegui Analisi di Scenario"):
        # Calcola budget minimo necessario
        min_budget_needed = 0
        for camp in campaigns:
            min_budget_needed += camp["cost"] * (min_share * total_leads)
        
        if budget_max_A < min_budget_needed:
            st.error(f"""
            Scenario A non ottimale: il budget assegnato ({int(budget_max_A):,} €) non è sufficiente.
            
            Con i vincoli attuali:
            - {total_leads:,} lead totali
            - {min_share:.1%} share minima per campagna
            - {corpo_percent:.1%} minimo lead corpo
            
            È necessario un budget di almeno {int(min_budget_needed):,} € oppure:
            1. Riduci il numero totale di lead
            2. Abbassa la percentuale minima per campagna
            3. Abbassa la percentuale minima di lead corpo
            """)
            return

        statusA, xA, profitA = solve_mip(
            campaigns, total_leads, corpo_percent, 
            min_share, budget_max_A, weight_immediate
        )
        if statusA != "Optimal":
            st.error(f"""
            Scenario A non ottimale. Status: {statusA}
            
            Questo potrebbe dipendere da:
            1. Budget insufficiente per i vincoli impostati
            2. Combinazione di vincoli troppo restrittiva
            
            Prova a:
            - Aumentare il budget
            - Ridurre il numero di lead richiesti
            - Abbassare le percentuali minime
            """)
            return

        dfA = compute_solution_df(campaigns, xA, weight_immediate)

        statusB, xB, profitB = solve_mip(
            campaigns, total_leads, corpo_percent, 
            min_share, 1e9, weight_immediate
        )
        if statusB != "Optimal":
            st.error(f"Scenario B non ottimale o infeasible. Status: {statusB}")
            return

        dfB = compute_solution_df(campaigns, xB, weight_immediate)

        st.write("## Risultati Scenario A (Budget limitato)")
        st.table(dfA)

        excelA, fnameA_xlsx = export_to_excel(dfA, "ScenarioA")
        pdfA, fnameA_pdf = export_to_pdf(dfA, "ScenarioA")

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

        st.write("## Risultati Scenario B (Budget illimitato)")
        st.table(dfB)

        excelB, fnameB_xlsx = export_to_excel(dfB, "ScenarioB")
        pdfB, fnameB_pdf = export_to_pdf(dfB, "ScenarioB")

        st.download_button(
    "Esporta Scenario B in Excel",
    excelB,
    fnameB_xlsx,
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
        st.download_button(
            label="Esporta Scenario B in PDF",
            data=pdfB,
            file_name=fnameB_pdf,
            mime="application/pdf"
        )

        st.subheader("Analisi di Scenario: Confronto A vs B")
        
        totA = dfA.loc["TOTALE"]
        totB = dfB.loc["TOTALE"]

        extra_cost = totB["Costo Tot"] - totA["Costo Tot"]
        extra_margin_imm = totB["Margine Immediato"] - totA["Margine Immediato"]
        extra_margin_60d = totB["Margine 60gg"] - totA["Margine 60gg"]
        extra_margin_w = totB["Margine Pesato"] - totA["Margine Pesato"]
        extra_leads = totB["Leads"] - totA["Leads"]

        st.markdown(f"""
        **Scenario A**:
        - Spesa: {int(totA["Costo Tot"]):,} €
        - Margine immediato: {int(totA["Margine Immediato"]):,} €
        - Margine 60gg: {int(totA["Margine 60gg"]):,} €
        - Margine pesato: {int(totA["Margine Pesato"]):,} €
        - Lead: {int(totA["Leads"]):,}

        **Scenario B**:
        - Spesa: {int(totB["Costo Tot"]):,} €
        - Margine immediato: {int(totB["Margine Immediato"]):,} €
        - Margine 60gg: {int(totB["Margine 60gg"]):,} €
        - Margine pesato: {int(totB["Margine Pesato"]):,} €
        - Lead: {int(totB["Leads"]):,}

        **Differenze (B - A)**:
        - Spesa extra: {int(extra_cost):,} €
        - Margine immediato extra: {int(extra_margin_imm):,} €
        - Margine 60gg extra: {int(extra_margin_60d):,} €
        - Margine pesato extra: {int(extra_margin_w):,} €
        - Lead extra: {int(extra_leads):,}
        """)

if __name__ == "__main__":
    main()