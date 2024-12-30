from anthropic import Anthropic
import streamlit as st

class CampaignAnalyzer:
    def __init__(self):
        """Inizializza l'analizzatore con il client Anthropic."""
        try:
            if "anthropic_api_key" not in st.secrets:
                raise ValueError("La chiave API Anthropic non è configurata nei secrets di Streamlit")
            
            self.client = Anthropic(api_key=st.secrets["anthropic_api_key"])
            #self.model = "claude-3-5-haiku-20241022"
            self.model = "claude-3-5-sonnet-20241022"
            self.max_tokens = 1024
            self.temperature = 0.75
        except Exception as e:
            st.error(f"Errore nell'inizializzazione del client Anthropic: {str(e)}")
            raise

    def generate_prompt(self, dfA, dfB):
        """Genera il prompt per l'analisi delle campagne."""
        prompt = f"""Analizza i seguenti dati di campagne marketing per due scenari (A: budget limitato, B: budget illimitato):

Scenario A (Budget Limitato):
{dfA.to_string()}

Scenario B (Budget Illimitato):
{dfB.to_string()}

Ricordati che stai lavorando come un direttore marketing con 20 anni di esperienza e stiamo analizzando campagne di digital marketing per la più grande catena di centri estetici italiani BIOLASER
Per favore:
1. Identifica le campagne più efficienti
2. Analizza il ROI incrementale tra scenario A e B
3. Suggerisci strategie di allocazione budget basate su:
   - Marginalità immediata vs 60gg
   - Performance per categoria
   - Costo per lead
4. Fornisci 3 raccomandazioni concrete per ottimizzare il budget

Concludi con un riassunto strategico di 3-4 paragrafi dal punto di vista di un Direttore Digital Marketing con 20 anni di esperienza, evidenziando:
- Non citare mai il tuo ruolo
- L'impatto delle performance sulla strategia complessiva dell'azienda
- Le opportunità di crescita identificate
- I rischi e le sfide da gestire


Considera eventuali vincoli operativi o di mercato nelle tue raccomandazioni.
Formatta la risposta in modo chiaro e strutturato."""
        return prompt

    def analyze_campaigns(self, dfA, dfB):
        """
        Analizza i dati delle campagne usando Anthropic API.
        
        Args:
            dfA (pd.DataFrame): DataFrame con i risultati dello scenario A
            dfB (pd.DataFrame): DataFrame con i risultati dello scenario B
            
        Returns:
            str: Analisi dettagliata delle campagne
        """
        try:
            prompt = self.generate_prompt(dfA, dfB)
            
            message = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )
            
            return message.content
            
        except Exception as e:
            st.error(f"Errore nell'analisi delle campagne: {str(e)}")
            return None

def display_analysis_section(dfA, dfB):
    """
    Aggiunge la sezione di analisi AI alla Streamlit app.
    Da chiamare dopo aver mostrato i risultati degli scenari.
    """
    st.write("---")
    st.subheader("Analisi AI delle Campagne")
    
    analyzer = CampaignAnalyzer()
    
    if st.button("Genera Analisi AI"):
        with st.spinner("Analisi in corso..."):
            analysis = analyzer.analyze_campaigns(dfA, dfB)
            if analysis:
                st.markdown("""
                ### Analisi delle Performance
                """)
                st.markdown(analysis)