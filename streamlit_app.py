import math

def solve_marketing_leads():
    """
    Esempio di soluzione 'manuale' (greedy+ricerca esaustiva su 1 dimensione)
    per distribuire i lead fra campagne laser e corpo.
    NON usa librerie esterne di ottimizzazione (come pulp/ortools).
    
    Viene chiesto all'utente:
      - Numero di campagne
      - Per ogni campagna: nome, categoria, costo per lead, ricavo per lead
      - Totale di lead
      - Percentuale minima di 'corpo'
    
    Restituisce (stampa) la soluzione con profitto massimo.
    
    Limitazioni:
      - Se in una categoria ci sono >=3 campagne, la distribuzione interna
        non è necessariamente ottima.
      - Il ciclo su tutti i possibili “lead_corpo” da min a max potrebbe
        essere costoso per numeri molto grandi.
    """
    
    print("=== Ottimizzatore 'manuale' senza librerie esterne ===\n")
    
    # 1. Input
    while True:
        try:
            n = int(input("Inserisci il numero di campagne (minimo 2, max 10): "))
            if 2 <= n <= 10:
                break
            else:
                print("Numero non valido.")
        except ValueError:
            print("Inserisci un intero valido.")

    campaigns = []
    for i in range(n):
        print(f"\n--- Campagna #{i+1} ---")
        name = input("Nome campagna: ").strip()
        category = ""
        while category not in ["laser", "corpo"]:
            category = input("Categoria (laser/corpo): ").lower().strip()
        
        while True:
            try:
                cost = float(input("Costo per lead: "))
                if cost >= 0:
                    break
                else:
                    print("Il costo per lead deve essere >= 0.")
            except ValueError:
                print("Inserisci un valore numerico.")
        
        while True:
            try:
                revenue = float(input("Ricavo per lead: "))
                if revenue >= 0:
                    break
                else:
                    print("Il ricavo per lead deve essere >= 0.")
            except ValueError:
                print("Inserisci un valore numerico.")
        
        net_profit = revenue - cost

        campaigns.append({
            "name": name if name else f"Camp_{i+1}",
            "category": category,
            "cost": cost,
            "revenue": revenue,
            "net_profit": net_profit
        })
    
    while True:
        try:
            total_leads = int(input("\nTotale dei lead da produrre (intero): "))
            if total_leads > 0:
                break
            else:
                print("Deve essere > 0.")
        except ValueError:
            print("Inserisci un intero valido.")
    
    while True:
        try:
            corpo_percent = float(input("Percentuale minima di 'corpo' (0..1): "))
            if 0 <= corpo_percent <= 1:
                break
            else:
                print("La percentuale deve essere compresa tra 0 e 1.")
        except ValueError:
            print("Inserisci un valore numerico (es. 0.33).")
    
    # 2. Suddividiamo le campagne per categoria
    laser_camps = [c for c in campaigns if c["category"] == "laser"]
    corpo_camps = [c for c in campaigns if c["category"] == "corpo"]
    
    # Funzione di supporto: alloca i leads totali di una singola categoria
    # tra le campagne di quella categoria, in modo "greedy semplificato":
    # - Se c'è solo 1 campagna, tutti i lead a quella campagna.
    # - Se ce ne sono >=2, assegniamo il 20% al min net_profit e il resto
    #   al max net_profit. (Ignora le intermedie se fossero più di 2.)
    def allocate_greedy(category_camps, leads_cat):
        """
        Restituisce una lista di (camp_name, leads_assigned, cost, revenue, profit).
        Se ci sono 3+ campagne, assegna ugualmente 20% a quella col net_profit min
        e 80% a quella col net_profit max, ignorando le altre.
        """
        if leads_cat <= 0:
            return [(c["name"], 0, 0, 0, 0) for c in category_camps]

        if len(category_camps) == 1:
            c = category_camps[0]
            assigned = leads_cat
            cost_tot = assigned * c["cost"]
            rev_tot = assigned * c["revenue"]
            prof_tot = assigned * c["net_profit"]
            return [(c["name"], assigned, cost_tot, rev_tot, prof_tot)]
        
        # Se >=2 campagne
        # 1) Trova la min e la max per net_profit
        c_min = min(category_camps, key=lambda x: x["net_profit"])
        c_max = max(category_camps, key=lambda x: x["net_profit"])
        
        # (grezzo) Assegniamo 20% a c_min, 80% a c_max
        min_leads = int(round(0.2 * leads_cat))
        max_leads = leads_cat - min_leads
        
        # Info c_min
        cost_min = min_leads * c_min["cost"]
        rev_min = min_leads * c_min["revenue"]
        prof_min = min_leads * c_min["net_profit"]
        
        # Info c_max
        cost_max = max_leads * c_max["cost"]
        rev_max = max_leads * c_max["revenue"]
        prof_max = max_leads * c_max["net_profit"]
        
        # Se ci sono altre campagne oltre min e max, le ignoriamo in questa logica
        # (potrebbe NON essere ottimale con 3+ campagne, ma è un esempio semplificato!)
        
        alloc_list = [
            (c_min["name"], min_leads, cost_min, rev_min, prof_min),
            (c_max["name"], max_leads, cost_max, rev_max, prof_max)
        ]
        
        # Per le altre campagne (se presenti), assegniamo 0
        for c in category_camps:
            if c["name"] not in [c_min["name"], c_max["name"]]:
                alloc_list.append((c["name"], 0, 0, 0, 0))
        
        return alloc_list
    
    # 3. Ricerchiamo la soluzione migliore iterando su quanti lead assegnare al "corpo"
    min_corpo_leads = int(math.ceil(corpo_percent * total_leads))  # almeno questa soglia
    best_solution = None
    best_profit = -1e9

    for corpo_leads_assigned in range(min_corpo_leads, total_leads+1):
        laser_leads_assigned = total_leads - corpo_leads_assigned
        
        # Alloco i leads per la categoria "corpo"
        corpo_alloc = allocate_greedy(corpo_camps, corpo_leads_assigned)
        # Alloco i leads per la categoria "laser"
        laser_alloc = allocate_greedy(laser_camps, laser_leads_assigned)
        
        # Sommiamo i profitti totali
        total_profit = sum(item[4] for item in corpo_alloc) + sum(item[4] for item in laser_alloc)
        
        if total_profit > best_profit:
            best_profit = total_profit
            best_solution = (corpo_alloc, laser_alloc, corpo_leads_assigned, laser_leads_assigned)
    
    # 4. Stampo la soluzione migliore trovata
    print("\n=== RISULTATI ===\n")
    if best_solution is None:
        print("Nessuna soluzione trovata (imprevisto).")
        return
    
    corpo_alloc, laser_alloc, corpo_leads_used, laser_leads_used = best_solution
    
    # Stampa assegnazione corpo
    if corpo_camps:
        print("--- CATEGORIA 'corpo' ---")
        for (camp_name, leads, cost_t, rev_t, prof_t) in sorted(corpo_alloc, key=lambda x: x[0]):
            print(f" Campagna: {camp_name}")
            print(f"   Leads: {leads}")
            print(f"   Costo Tot: {int(round(cost_t))}")
            print(f"   RicavoTot: {int(round(rev_t))}")
            print(f"   Margine:   {int(round(prof_t))}")
            print("")

    # Stampa assegnazione laser
    if laser_camps:
        print("--- CATEGORIA 'laser' ---")
        for (camp_name, leads, cost_t, rev_t, prof_t) in sorted(laser_alloc, key=lambda x: x[0]):
            print(f" Campagna: {camp_name}")
            print(f"   Leads: {leads}")
            print(f"   Costo Tot: {int(round(cost_t))}")
            print(f"   RicavoTot: {int(round(rev_t))}")
            print(f"   Margine:   {int(round(prof_t))}")
            print("")

    # Riepilogo finale
    tot_corpo_profit = sum(item[4] for item in corpo_alloc)
    tot_laser_profit = sum(item[4] for item in laser_alloc)
    tot_cost = sum(item[2] for item in corpo_alloc) + sum(item[2] for item in laser_alloc)
    tot_revenue = sum(item[3] for item in corpo_alloc) + sum(item[3] for item in laser_alloc)
    tot_margin = tot_corpo_profit + tot_laser_profit

    print(f"Totale lead: {total_leads}")
    print(f"Lead CORPO: {corpo_leads_used} (>= {int(round(corpo_percent*100))}% del totale)")
    print(f"Lead LASER: {laser_leads_used}")
    print(f"\nCosto totale:   {int(round(tot_cost))}")
    print(f"Ricavo totale:  {int(round(tot_revenue))}")
    print(f"Profitto totale: {int(round(tot_margin))}")


if __name__ == "__main__":
    solve_marketing_leads()
