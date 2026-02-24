import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from nicegui import ui

# CSV apdorojimas
def paruosti_duomenis():
    print('atidaromas CSV ir valomi duomenys')
    df_raw = pd.read_csv('dataset/medium_articles.csv')
    df_raw = df_raw.dropna(subset=['text', 'title']).reset_index(drop=True)

    # blacklistas
    bad_words = {
        'medium', 'learn', 'using', 'things', 'function', 'model'
    }

    spam_text = "Learn more. Medium is an open platform"

    def is_clean_english(row):
        text = str(row['text'])
        if spam_text in text or len(text) < 500: return False
        if re.search(r'[\u4e00-\u9fff]', text): return False
        ascii_chars = len(text.encode('ascii', 'ignore'))
        return (ascii_chars / len(text)) > 0.85 if len(text) > 0 else False

    df_clean = df_raw[df_raw.apply(is_clean_english, axis=1)].copy()
    df_sample = df_clean.sample(min(100000, len(df_clean))).copy()

    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000, ngram_range=(1, 1))
    X = vectorizer.fit_transform(df_sample['text'])

    n_clusters = 5
    kmeans = KMeans(n_clusters=n_clusters, n_init=10)
    df_sample['tema_id'] = kmeans.fit_predict(X)

    terms = vectorizer.get_feature_names_out()
    order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]

    temos_info = {}
    panaudoti_pavadinimai = set()

    for i in range(n_clusters):
        top_words = [terms[ind] for ind in order_centroids[i, :20]] # turi but nesikartojantys
        
        # filtravimas
        parinktas_pavad = f"Topic {i+1}"
        for word in top_words:
            w_title = word.title()
            if len(word) > 3 and word.lower() not in bad_words and w_title not in panaudoti_pavadinimai:
                parinktas_pavad = w_title
                panaudoti_pavadinimai.add(w_title)
                break 

        temos_info[i] = {
            'pavadinimas': parinktas_pavad,
            'raktazodziai': ', '.join(top_words[:10])
        }
    
    return df_sample, temos_info

# tam, kad duomenis nebutu paruosiami is naujo parefreshinus
if __name__ in {"__main__", "__mp_main__"}:
    df_processed, info_processed = paruosti_duomenis()

# UI logika visa (didele dalis vibecodinta)

class StraipsniuNarsykle:
    def __init__(self, data, temos):
        self.data = data
        self.temos = temos
        self.pasirinkta_tema = None
        self.paieskos_tekstas = ''
        self.puslapis = 1
        self.per_puslapi = 7

    @property
    def filtruoti_duomenys(self):
        d = self.data
        if self.pasirinkta_tema is not None:
            d = d[d['tema_id'] == self.pasirinkta_tema]
        if self.paieskos_tekstas:
            mask = d['title'].str.contains(self.paieskos_tekstas, case=False, na=False) | \
                   d['text'].str.contains(self.paieskos_tekstas, case=False, na=False)
            d = d[mask]
        return d

    def rodyti_straipsni(self, title, full_text):
        with ui.dialog() as dialog, ui.card().classes('w-[90vw] max-w-4xl h-[80vh]'):
            with ui.column().classes('w-full h-full p-4'):
                ui.label(title).classes('text-2xl font-bold mb-2')
                ui.separator()
                with ui.scroll_area().classes('flex-grow mt-4 border p-4 rounded bg-gray-50 dark:bg-slate-900'):
                    ui.label(str(full_text)).classes('text-base leading-relaxed whitespace-pre-wrap')
                with ui.row().classes('w-full justify-end mt-4'):
                    ui.button('Uždaryti', on_click=dialog.close).props('outline')
        dialog.open()

    def rodyti_straipsnius(self):
        rezultatai_container.clear()
        d = self.filtruoti_duomenys
        start = (self.puslapis - 1) * self.per_puslapi
        end = start + self.per_puslapi
        puslapio_duomenys = d.iloc[start:end]

        with rezultatai_container:
            if d.empty:
                ui.label('Straipsnių nerasta.').classes('mt-10 opacity-50 text-center w-full')
                return
            
            ui.label(f'Rasta: {len(d)}').classes('text-xs opacity-50 mb-2')
            for _, row in puslapio_duomenys.iterrows():
                with ui.card().classes('w-full mb-4 shadow hover:shadow-md transition-all'):
                    with ui.card_section():
                        ui.label(row['title']).classes('text-xl font-bold text-blue-600 dark:text-blue-400')
                        ui.label(str(row['text'])[:350] + '...').classes('text-sm mb-4')
                        ui.button('Skaityti daugiau', on_click=lambda r=row: self.rodyti_straipsni(r['title'], r['text'])).props('flat icon=menu_book')

            # paginationas
            max_p = max(1, (len(d) - 1) // self.per_puslapi + 1)
            with ui.row().classes('w-full justify-center items-center gap-4 mt-4 pb-10'):
                ui.button(icon='arrow_back', on_click=lambda: self.keisti_puslapi(-1)).set_enabled(self.puslapis > 1)
                ui.label(f'{self.puslapis} / {max_p}').classes('font-bold')
                ui.button(icon='arrow_forward', on_click=lambda: self.keisti_puslapi(1)).set_enabled(self.puslapis < max_p)

    def keisti_puslapi(self, delta):
        self.puslapis += delta
        self.rodyti_straipsnius()
        ui.run_javascript('window.scrollTo(0, 0);')

    def nustatyti_tema(self, tema_id):
        self.pasirinkta_tema = tema_id
        self.puslapis = 1
        self.rodyti_straipsnius()

# nicegui pagrindinis puslapis

@ui.page('/')
def main_page():
    narsykle = StraipsniuNarsykle(df_processed, info_processed)
    dark = ui.dark_mode()

    with ui.header().classes('bg-slate-800 items-center justify-between p-4'):
        ui.label('AI LAB WORK 1: NLP ON ARTICLES').classes('text-xl font-black text-white')
        ui.button(icon='dark_mode', on_click=dark.toggle).props('flat color=white')

    with ui.column().classes('w-full max-w-5xl mx-auto p-4'):
        ui.input(placeholder='Paieška...', on_change=lambda e: narsykle.ieskoti_su_reiksme(e.value)) \
            .classes('w-full mb-6').props('standout rounded icon=search')
        
        # temu mygtukai
        with ui.row().classes('w-full gap-2 mb-8'):
            ui.button('Visi', on_click=lambda: narsykle.nustatyti_tema(None)).props('rounded outline')
            for tema_id, info in info_processed.items():
                btn = ui.button(info['pavadinimas'], on_click=lambda t=tema_id: narsykle.nustatyti_tema(t)) \
                    .props('rounded elevated color=indigo')
                with btn:
                    ui.tooltip(f"Raktažodžiai: {info['raktazodziai']}")

        global rezultatai_container
        rezultatai_container = ui.column().classes('w-full')
        
        # papildomas metodas narsyklei, kad veiktų su input
        narsykle.ieskoti_su_reiksme = lambda v: [setattr(narsykle, 'paieskos_tekstas', v), 
                                                setattr(narsykle, 'puslapis', 1), 
                                                narsykle.rodyti_straipsnius()]
        
        narsykle.rodyti_straipsnius()

ui.run(title='Article NLP', port=8080, reload=True) 