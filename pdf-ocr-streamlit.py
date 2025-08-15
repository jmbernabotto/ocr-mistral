import streamlit as st
import os
import tempfile
import requests
from pathlib import Path
from mistralai import Mistral
from typing import Dict, Any, List
import fitz  # PyMuPDF
import time
import base64


class PDFOCRProcessor:
    """
    Application Streamlit pour OCR de PDFs avec Mistral AI et enrichissement
    """
    
    def __init__(self):
        self.ocr_model = "mistral-ocr-latest"
        
        # Initialisation de la session state
        if 'documents' not in st.session_state:
            st.session_state.documents = []
        if 'client' not in st.session_state:
            st.session_state.client = None
        if 'api_key_valid' not in st.session_state:
            st.session_state.api_key_valid = False
    
    def setup_api_key(self):
        """Configuration de la clé API"""
        st.sidebar.header("🔑 Configuration API")
        
        # Input pour la clé API
        api_key = st.sidebar.text_input(
            "Clé API Mistral",
            type="password",
            value=os.environ.get("MISTRAL_API_KEY", ""),
            help="Entrez votre clé API Mistral"
        )
        
        if api_key:
            if st.session_state.client is None or not st.session_state.api_key_valid:
                try:
                    # Test de la clé API
                    client = Mistral(api_key=api_key)
                    models = client.models.list()
                    
                    st.session_state.client = client
                    st.session_state.api_key_valid = True
                    st.sidebar.success("✅ API connectée avec succès!")
                    
                except Exception as e:
                    st.sidebar.error(f"❌ Erreur API: {str(e)}")
                    st.session_state.api_key_valid = False
            else:
                st.sidebar.success("✅ API connectée")
        else:
            st.sidebar.warning("⚠️ Clé API requise")
            st.session_state.api_key_valid = False
        
        return st.session_state.api_key_valid
    
    def process_pdf_ocr(self, file_bytes: bytes, file_name: str) -> Dict[str, Any]:
        """Traite un PDF avec OCR Mistral en utilisant l'upload direct"""
        try:
            # Upload du PDF vers Mistral
            uploaded_file = st.session_state.client.files.upload(
                file={
                    "file_name": file_name,
                    "content": file_bytes,
                },
                purpose="ocr"
            )
            
            # Obtenir l'URL signée
            signed_url_response = st.session_state.client.files.get_signed_url(
                file_id=uploaded_file.id
            )
            
            # OCR avec Mistral en utilisant l'URL signée
            ocr_response = st.session_state.client.ocr.process(
                model=self.ocr_model,
                document={
                    "type": "document_url",
                    "document_url": signed_url_response.url
                }
            )
            
            # Extraction du texte par page sous forme de dictionnaire index -> texte
            page_texts_map = {page.index: page.markdown for page in ocr_response.pages}
            
            # Reconstruire le texte complet dans l'ordre des pages pour l'affichage/téléchargement
            full_text_list = []
            if page_texts_map:
                max_index = max(page_texts_map.keys())
                full_text_list = [page_texts_map.get(i, "") for i in range(max_index + 1)]
            extracted_text = "\n\n---\n\n".join(full_text_list)
            
            return {
                'name': file_name,
                'type': 'PDF',
                'original_bytes': file_bytes,
                'extracted_text': extracted_text,
                'page_texts_map': page_texts_map,
                'status': 'success'
            }
            
        except Exception as e:
            st.error(f"Erreur lors du traitement OCR: {str(e)}")
            return {
                'name': file_name,
                'type': 'PDF',
                'status': 'error',
                'error': str(e)
            }
    
    def add_text_to_pdf(self, pdf_bytes: bytes, page_texts_map: dict, file_name: str) -> bytes:
        """Crée un PDF 'sandwich' en superposant le texte extrait de manière invisible"""
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            num_pages = doc.page_count
            
            # Aligner les textes extraits sur les pages réelles du PDF
            page_texts = [page_texts_map.get(i, "") for i in range(num_pages)]
            
            for i in range(num_pages):
                page = doc[i]
                text = page_texts[i]
                
                if not text.strip():
                    continue
                
                # OPTION 3: La méthode la plus robuste.
                # Insère tout le texte de la page avec une police minuscule dans un coin.
                # C'est invisible, donc la mise en page n'importe pas, mais le contenu est
                # entièrement présent pour la recherche, sans aucun risque de troncature.
                page.insert_text(
                    point=fitz.Point(5, 5),  # Coin supérieur gauche
                    text=text,
                    fontsize=1,          # Police minuscule pour garantir que tout rentre
                    render_mode=3,       # Mode crucial pour l'invisibilité
                    fill_opacity=0       # Opacité du remplissage à 0
                )
                
            pdf_bytes_enriched = doc.write()
            doc.close()
            return pdf_bytes_enriched
            
        except Exception as e:
            st.error(f"Erreur lors de la création du PDF sandwich: {str(e)}")
            return pdf_bytes
    
    def upload_documents(self):
        """Interface d'upload de documents"""
        st.header("📚 Upload de PDFs")
        
        # Upload multiple files (PDF uniquement)
        uploaded_files = st.file_uploader(
            "Choisissez vos PDFs à traiter",
            type=['pdf'],
            accept_multiple_files=True,
            help="Sélectionnez un ou plusieurs fichiers PDF"
        )
        
        if uploaded_files:
            # Bouton pour traiter les fichiers
            if st.button("🔄 Traiter les PDFs avec OCR", type="primary"):
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                processed_count = 0
                total_files = len(uploaded_files)
                
                for i, uploaded_file in enumerate(uploaded_files):
                    # Vérifier si le document n'est pas déjà chargé
                    if uploaded_file.name in [doc['name'] for doc in st.session_state.documents]:
                        status_text.text(f"⚠️ {uploaded_file.name} déjà chargé, ignoré")
                        continue
                    
                    status_text.text(f"📄 OCR de {uploaded_file.name}...")
                    
                    # Lire le fichier
                    file_bytes = uploaded_file.read()
                    
                    # Traitement OCR
                    result = self.process_pdf_ocr(file_bytes, uploaded_file.name)
                    
                    # Ajouter à la session
                    if result['status'] == 'success':
                        st.session_state.documents.append(result)
                        processed_count += 1
                        status_text.text(f"✅ {uploaded_file.name} traité avec succès")
                    else:
                        st.error(f"❌ Erreur avec {uploaded_file.name}: {result.get('error', 'Erreur inconnue')}")
                    
                    # Mise à jour de la barre de progression
                    progress_bar.progress((i + 1) / total_files)
                    time.sleep(0.5)  # Pause pour l'UX
                
                status_text.text(f"🎉 Traitement terminé! {processed_count}/{total_files} PDFs traités")
                time.sleep(2)
                st.rerun()
    
    def show_document_collection(self):
        """Affiche la collection de documents"""
        if not st.session_state.documents:
            st.info("📝 Aucun PDF chargé. Utilisez l'onglet 'Upload' pour ajouter des PDFs.")
            return
        
        st.header(f"📊 Collection ({len(st.session_state.documents)} PDFs)")
        
        # Affichage des documents avec possibilité de suppression
        for i, doc in enumerate(st.session_state.documents):
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.write(f"📄 **{doc['name']}** ({doc['type']})")
                if doc.get('extracted_text'):
                    with st.expander("Voir le texte extrait"):
                        st.text(doc['extracted_text'][:500] + "..." if len(doc['extracted_text']) > 500 else doc['extracted_text'])
            
            with col2:
                st.write(f"*{doc['type']}*")
            
            with col3:
                if st.button(f"🗑️", key=f"delete_{i}", help=f"Supprimer {doc['name']}"):
                    st.session_state.documents.pop(i)
                    st.rerun()
        
        # Bouton pour tout supprimer
        if st.session_state.documents:
            st.divider()
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                if st.button("🗑️ Tout supprimer", type="secondary"):
                    st.session_state.documents = []
                    st.rerun()
    
    def download_section(self):
        """Section de téléchargement des PDFs enrichis"""
        if not st.session_state.documents:
            st.warning("⚠️ Veuillez d'abord charger des PDFs dans l'onglet 'Upload'")
            return
        
        st.header("⬇️ Télécharger les PDFs enrichis")
        
        # Traitement et téléchargement de chaque PDF
        for i, doc in enumerate(st.session_state.documents):
            if doc.get('status') == 'success' and doc.get('page_texts_map'):
                
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.write(f"📄 **{doc['name']}**")
                    st.write(f"Texte extrait: {len(doc['extracted_text'])} caractères")
                
                with col2:
                    # Préparer le PDF enrichi
                    enriched_pdf_bytes = self.add_text_to_pdf(
                        doc['original_bytes'],
                        doc.get('page_texts_map', {}),
                        doc['name']
                    )
                    
                    # Préparer le nom du fichier
                    base_name = Path(doc['name']).stem
                    enriched_name = f"{base_name}_enrichi.pdf"
                    
                    # Bouton de téléchargement direct
                    st.download_button(
                        label="💾 Télécharger",
                        data=enriched_pdf_bytes,
                        file_name=enriched_name,
                        mime="application/pdf",
                        key=f"download_file_{i}"
                    )
        
        # Section pour télécharger le texte extrait
        st.divider()
        st.subheader("📝 Télécharger le texte extrait (Markdown)")
        
        for i, doc in enumerate(st.session_state.documents):
            if doc.get('status') == 'success' and doc.get('extracted_text'):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.write(f"📄 **{doc['name']}** - Texte extrait")
                
                with col2:
                    # Bouton pour télécharger le texte en Markdown
                    base_name = Path(doc['name']).stem
                    md_name = f"{base_name}_texte.md"
                    
                    st.download_button(
                        label="📄 Télécharger .md",
                        data=doc['extracted_text'],
                        file_name=md_name,
                        mime="text/markdown",
                        key=f"download_text_{i}"
                    )
    
    def run(self):
        """Lance l'application Streamlit"""
        st.set_page_config(
            page_title="PDF OCR Processor",
            page_icon="📄",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        st.title("📄 PDF OCR Processor")
        st.markdown("*Extraction de texte et enrichissement de PDFs avec Mistral AI*")
        
        # Configuration API dans la sidebar
        api_ready = self.setup_api_key()
        
        if not api_ready:
            st.warning("🔑 Veuillez configurer votre clé API Mistral dans la barre latérale pour commencer.")
            st.info("Vous pouvez obtenir votre clé API sur [console.mistral.ai](https://console.mistral.ai)")
            return
        
        # Sidebar avec statistiques
        st.sidebar.header("📊 Statistiques")
        st.sidebar.metric("PDFs chargés", len(st.session_state.documents))
        
        # Onglets principaux
        tab1, tab2, tab3 = st.tabs(["📤 Upload", "📊 Documents", "⬇️ Télécharger"])
        
        with tab1:
            self.upload_documents()
        
        with tab2:
            self.show_document_collection()
        
        with tab3:
            self.download_section()


def main():
    """Point d'entrée principal"""
    app = PDFOCRProcessor()
    app.run()


if __name__ == "__main__":
    main() 