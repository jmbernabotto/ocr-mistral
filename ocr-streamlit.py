import streamlit as st
import os
import base64
from pathlib import Path
from mistralai import Mistral
from typing import Optional, Dict, Any, List
import mimetypes
import tempfile
import time


class StreamlitMultiDocChat:
    """
    Application Streamlit pour chat multi-documents avec Mistral AI
    """
    
    def __init__(self):
        self.model = "mistral-small-latest"
        self.ocr_model = "mistral-ocr-latest"
        
        # Initialisation de la session state
        if 'documents' not in st.session_state:
            st.session_state.documents = []
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
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
    
    def encode_image(self, image_bytes: bytes) -> str:
        """Encode une image en base64"""
        return base64.b64encode(image_bytes).decode('utf-8')
    
    def get_image_mime_type(self, file_name: str) -> str:
        """Détermine le type MIME d'une image"""
        ext = Path(file_name).suffix.lower()
        mime_map = {
            '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
            '.png': 'image/png', '.gif': 'image/gif',
            '.bmp': 'image/bmp', '.tiff': 'image/tiff',
            '.webp': 'image/webp'
        }
        return mime_map.get(ext, 'image/jpeg')
    
    def process_pdf(self, file_bytes: bytes, file_name: str) -> Dict[str, Any]:
        """Traite un fichier PDF"""
        try:
            uploaded_pdf = st.session_state.client.files.upload(
                file={
                    "file_name": file_name,
                    "content": file_bytes,
                },
                purpose="ocr"
            )
            
            signed_url_response = st.session_state.client.files.get_signed_url(
                file_id=uploaded_pdf.id
            )
            
            return {
                'name': file_name,
                'type': 'PDF',
                'file_id': uploaded_pdf.id,
                'signed_url': signed_url_response.url,
                'status': 'success'
            }
            
        except Exception as e:
            return {
                'name': file_name,
                'type': 'PDF',
                'status': 'error',
                'error': str(e)
            }
    
    def process_image(self, file_bytes: bytes, file_name: str) -> Dict[str, Any]:
        """Traite un fichier image"""
        try:
            base64_image = self.encode_image(file_bytes)
            mime_type = self.get_image_mime_type(file_name)
            
            ocr_response = st.session_state.client.ocr.process(
                model=self.ocr_model,
                document={
                    "type": "image_url",
                    "image_url": f"data:{mime_type};base64,{base64_image}"
                },
                include_image_base64=True
            )
            
            # Extraction du texte
            extracted_text = ""
            if hasattr(ocr_response, 'text'):
                extracted_text = ocr_response.text
            elif isinstance(ocr_response, dict) and 'text' in ocr_response:
                extracted_text = ocr_response['text']
            
            return {
                'name': file_name,
                'type': 'Image',
                'ocr_results': ocr_response,
                'extracted_text': extracted_text,
                'status': 'success'
            }
            
        except Exception as e:
            return {
                'name': file_name,
                'type': 'Image',
                'status': 'error',
                'error': str(e)
            }
    
    def upload_documents(self):
        """Interface d'upload de documents"""
        st.header("📚 Gestion des Documents")
        
        # Upload multiple files
        uploaded_files = st.file_uploader(
            "Choisissez vos documents à analyser",
            type=['pdf', 'jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff', 'webp'],
            accept_multiple_files=True,
            help="Vous pouvez sélectionner plusieurs fichiers PDF et images"
        )
        
        if uploaded_files:
            # Bouton pour traiter les fichiers
            if st.button("🔄 Traiter les documents", type="primary"):
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                processed_count = 0
                total_files = len(uploaded_files)
                
                for i, uploaded_file in enumerate(uploaded_files):
                    # Vérifier si le document n'est pas déjà chargé
                    if uploaded_file.name in [doc['name'] for doc in st.session_state.documents]:
                        status_text.text(f"⚠️ {uploaded_file.name} déjà chargé, ignoré")
                        continue
                    
                    status_text.text(f"📄 Traitement de {uploaded_file.name}...")
                    
                    # Lire le fichier
                    file_bytes = uploaded_file.read()
                    
                    # Traitement selon le type
                    if uploaded_file.name.lower().endswith('.pdf'):
                        result = self.process_pdf(file_bytes, uploaded_file.name)
                    else:
                        result = self.process_image(file_bytes, uploaded_file.name)
                    
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
                
                status_text.text(f"🎉 Traitement terminé! {processed_count}/{total_files} documents ajoutés")
                time.sleep(2)
                st.rerun()
    
    def show_document_collection(self):
        """Affiche la collection de documents"""
        if not st.session_state.documents:
            st.info("📝 Aucun document chargé. Utilisez l'onglet 'Upload' pour ajouter des documents.")
            return
        
        st.header(f"📊 Collection ({len(st.session_state.documents)} documents)")
        
        # Affichage des documents avec possibilité de suppression
        for i, doc in enumerate(st.session_state.documents):
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                if doc['type'] == 'PDF':
                    st.write(f"📄 **{doc['name']}** ({doc['type']})")
                else:
                    st.write(f"🖼️ **{doc['name']}** ({doc['type']})")
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
                    st.session_state.chat_history = []
                    st.rerun()
    
    def chat_interface(self):
        """Interface de chat"""
        if not st.session_state.documents:
            st.warning("⚠️ Veuillez d'abord charger des documents dans l'onglet 'Upload'")
            return
        
        st.header(f"💬 Chat avec {len(st.session_state.documents)} documents")
        
        # Affichage de l'historique du chat
        chat_container = st.container()
        
        with chat_container:
            for i, (question, response) in enumerate(st.session_state.chat_history):
                st.write(f"**🙋 Vous:** {question}")
                st.write(f"**🤖 Assistant:** {response}")
                st.divider()
        
        # Interface de saisie
        with st.form("chat_form", clear_on_submit=True):
            question = st.text_area(
                "Posez votre question sur les documents:",
                placeholder="Ex: Compare les budgets de ces documents, Quels sont les points communs ?, Résume l'ensemble...",
                height=100
            )
            
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                submit_button = st.form_submit_button("🚀 Envoyer", type="primary")
        
        # Traitement de la question
        if submit_button and question.strip():
            with st.spinner("🤖 Analyse des documents en cours..."):
                response = self.process_question(question)
                
                # Ajouter à l'historique
                st.session_state.chat_history.append((question, response))
                st.rerun()
    
    def process_question(self, question: str) -> str:
        """Traite une question sur tous les documents"""
        try:
            # Construction du message avec tous les documents
            content_parts = [{"type": "text", "text": question}]
            
            # Contexte textuel pour les images
            image_context = ""
            
            # Ajout des documents PDF (URLs signées)
            for doc in st.session_state.documents:
                if doc['type'] == 'PDF' and 'signed_url' in doc:
                    content_parts.append({
                        "type": "document_url",
                        "document_url": doc['signed_url']
                    })
                elif doc['type'] == 'Image' and 'extracted_text' in doc:
                    image_context += f"\n--- Contenu de {doc['name']} ---\n{doc['extracted_text']}\n"
            
            # Si on a des images, ajouter leur contenu textuel au message
            if image_context:
                enhanced_question = f"Voici le contenu extrait d'images par OCR :{image_context}\n\nQuestion portant sur tous les documents : {question}"
                content_parts[0]["text"] = enhanced_question
            
            messages = [{
                "role": "user",
                "content": content_parts
            }]
            
            chat_response = st.session_state.client.chat.complete(
                model=self.model,
                messages=messages
            )
            
            return chat_response.choices[0].message.content
            
        except Exception as e:
            return f"❌ Erreur lors du traitement : {str(e)}"
    
    def examples_section(self):
        """Section avec des exemples de questions"""
        st.header("💡 Exemples de Questions")
        
        st.write("Voici quelques exemples de questions que vous pouvez poser :")
        
        examples = [
            "📊 **Comparaison**: Compare les budgets mentionnés dans ces documents",
            "🔍 **Recherche**: Dans quels documents parle-t-on de dates limites ?",
            "📝 **Synthèse**: Résume les points clés de tous ces documents",
            "🔢 **Chiffres**: Quels sont tous les montants mentionnés ?",
            "👥 **Personnes**: Quelles personnes sont mentionnées dans l'ensemble ?",
            "📋 **Différences**: Quelles sont les différences entre ces contrats ?",
            "⚠️ **Contradictions**: Y a-t-il des contradictions entre ces documents ?",
            "📈 **Tendances**: Quelles tendances peut-on observer ?"
        ]
        
        for example in examples:
            st.write(example)
        
        st.info("💡 **Astuce**: Plus vous chargez de documents, plus l'analyse croisée sera riche!")
    
    def run(self):
        """Lance l'application Streamlit"""
        st.set_page_config(
            page_title="Multi-Doc Chat AI",
            page_icon="📚",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        st.title("📚 Multi-Document AI Chat")
        st.markdown("*Analysez et croisez plusieurs documents avec l'IA Mistral*")
        
        # Configuration API dans la sidebar
        api_ready = self.setup_api_key()
        
        if not api_ready:
            st.warning("🔑 Veuillez configurer votre clé API Mistral dans la barre latérale pour commencer.")
            st.info("Vous pouvez obtenir votre clé API sur [console.mistral.ai](https://console.mistral.ai)")
            return
        
        # Sidebar avec statistiques
        st.sidebar.header("📊 Statistiques")
        st.sidebar.metric("Documents chargés", len(st.session_state.documents))
        st.sidebar.metric("Questions posées", len(st.session_state.chat_history))
        
        # Onglets principaux
        tab1, tab2, tab3, tab4 = st.tabs(["📤 Upload", "📊 Documents", "💬 Chat", "💡 Exemples"])
        
        with tab1:
            self.upload_documents()
        
        with tab2:
            self.show_document_collection()
        
        with tab3:
            self.chat_interface()
        
        with tab4:
            self.examples_section()


def main():
    """Point d'entrée principal"""
    app = StreamlitMultiDocChat()
    app.run()


if __name__ == "__main__":
    main()