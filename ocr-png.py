import streamlit as st
import base64
import os
import zipfile
import io
from pathlib import Path
from mistralai import Mistral
import mimetypes
from typing import List, Dict, Any
import pandas as pd
from datetime import datetime
import json

class StreamlitOCRProcessor:
    """
    Application Streamlit pour l'OCR de documents en masse
    """
    
    def __init__(self):
        self.ocr_model = "mistral-ocr-latest"
        self.supported_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp'}
        
    def get_mistral_client(self, api_key: str) -> Mistral:
        """Initialise le client Mistral avec la clé API"""
        return Mistral(api_key=api_key)
    
    def encode_image(self, image_bytes: bytes) -> str:
        """Encode une image en base64"""
        return base64.b64encode(image_bytes).decode('utf-8')
    
    def get_image_mime_type(self, filename: str) -> str:
        """Détermine le type MIME d'une image"""
        mime_type, _ = mimetypes.guess_type(filename)
        if mime_type and mime_type.startswith('image/'):
            return mime_type
        
        ext = Path(filename).suffix.lower()
        mime_map = {
            '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
            '.png': 'image/png', '.gif': 'image/gif',
            '.bmp': 'image/bmp', '.tiff': 'image/tiff',
            '.webp': 'image/webp'
        }
        return mime_map.get(ext, 'image/jpeg')
    
    def process_single_image(self, client: Mistral, image_bytes: bytes, filename: str) -> Dict[str, Any]:
        """Traite une seule image avec OCR"""
        try:
            base64_image = self.encode_image(image_bytes)
            mime_type = self.get_image_mime_type(filename)
            
            ocr_response = client.ocr.process(
                model=self.ocr_model,
                document={
                    "type": "image_url",
                    "image_url": f"data:{mime_type};base64,{base64_image}"
                },
                include_image_base64=False  # Économise de la bande passante
            )
            
            # Extraction du texte selon la structure de réponse
            extracted_text = ""
            if hasattr(ocr_response, 'text'):
                extracted_text = ocr_response.text
            elif isinstance(ocr_response, dict):
                if 'text' in ocr_response:
                    extracted_text = ocr_response['text']
                elif 'choices' in ocr_response and len(ocr_response['choices']) > 0:
                    extracted_text = ocr_response['choices'][0].get('message', {}).get('content', '')
            
            return {
                'filename': filename,
                'status': 'success',
                'text': extracted_text,
                'error': None,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'filename': filename,
                'status': 'error',
                'text': '',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def create_results_zip(self, results: List[Dict[str, Any]]) -> bytes:
        """Crée un fichier ZIP avec les résultats"""
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Fichier CSV avec résumé
            df = pd.DataFrame(results)
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            zip_file.writestr('ocr_summary.csv', csv_buffer.getvalue())
            
            # Fichiers texte individuels
            for result in results:
                if result['status'] == 'success' and result['text'].strip():
                    filename = Path(result['filename']).stem + '.txt'
                    zip_file.writestr(f'extracted_texts/{filename}', result['text'])
            
            # Fichier JSON avec tous les détails
            zip_file.writestr('ocr_results.json', json.dumps(results, indent=2, ensure_ascii=False))
        
        zip_buffer.seek(0)
        return zip_buffer.getvalue()

def main():
    st.set_page_config(
        page_title="OCR en Masse - Mistral AI",
        page_icon="📄",
        layout="wide"
    )
    
    st.title("🤖 OCR en Masse avec Mistral AI")
    st.markdown("**Extrayez le texte de jusqu'à 200 images PNG simultanément**")
    
    processor = StreamlitOCRProcessor()
    
    # Sidebar pour la configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Clé API
        api_key = st.text_input(
            "Clé API Mistral",
            type="password",
            help="Votre clé API Mistral AI"
        )
        
        if not api_key:
            st.warning("Veuillez entrer votre clé API Mistral pour continuer")
            st.info("Vous pouvez obtenir votre clé API sur [console.mistral.ai](https://console.mistral.ai)")
    
    # Interface principale
    if api_key:
        try:
            client = processor.get_mistral_client(api_key)
            
            # Upload des fichiers
            st.header("📁 Upload des Images")
            uploaded_files = st.file_uploader(
                "Sélectionnez vos images PNG (max 200 fichiers)",
                type=['png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'webp'],
                accept_multiple_files=True,
                help="Formats supportés: PNG, JPG, JPEG, GIF, BMP, TIFF, WEBP"
            )
            
            if uploaded_files:
                # Validation du nombre de fichiers
                if len(uploaded_files) > 200:
                    st.error(f"Trop de fichiers sélectionnés ({len(uploaded_files)}). Maximum: 200 fichiers.")
                    return
                
                st.success(f"✅ {len(uploaded_files)} fichier(s) sélectionné(s)")
                
                # Aperçu des fichiers
                with st.expander("📋 Aperçu des fichiers"):
                    for i, file in enumerate(uploaded_files[:10]):  # Afficher max 10
                        st.write(f"{i+1}. {file.name} ({file.size:,} bytes)")
                    
                    if len(uploaded_files) > 10:
                        st.write(f"... et {len(uploaded_files) - 10} autres fichiers")
                
                # Bouton de traitement
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    process_button = st.button(
                        "🚀 Lancer l'OCR",
                        type="primary",
                        use_container_width=True
                    )
                
                if process_button:
                    # Barre de progression
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    results = []
                    
                    # Traitement des fichiers
                    for i, uploaded_file in enumerate(uploaded_files):
                        # Mise à jour de l'interface
                        progress = (i + 1) / len(uploaded_files)
                        progress_bar.progress(progress)
                        status_text.text(f"Traitement: {uploaded_file.name} ({i+1}/{len(uploaded_files)})")
                        
                        # Lecture du fichier
                        image_bytes = uploaded_file.read()
                        
                        # Traitement OCR
                        result = processor.process_single_image(
                            client, 
                            image_bytes, 
                            uploaded_file.name
                        )
                        results.append(result)
                        
                        # Reset du pointeur de fichier pour éviter les erreurs
                        uploaded_file.seek(0)
                    
                    # Finalisation
                    progress_bar.progress(1.0)
                    status_text.text("✅ Traitement terminé!")
                    
                    # Affichage des résultats
                    st.header("📊 Résultats")
                    
                    # Statistiques
                    success_count = sum(1 for r in results if r['status'] == 'success')
                    error_count = len(results) - success_count
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("✅ Succès", success_count)
                    with col2:
                        st.metric("❌ Erreurs", error_count)
                    with col3:
                        st.metric("📄 Total", len(results))
                    
                    # Tableau des résultats
                    df_results = pd.DataFrame(results)
                    
                    # Onglets pour différentes vues
                    tab1, tab2, tab3 = st.tabs(["📋 Résumé", "✅ Succès", "❌ Erreurs"])
                    
                    with tab1:
                        st.dataframe(
                            df_results[['filename', 'status', 'timestamp']],
                            use_container_width=True
                        )
                    
                    with tab2:
                        success_df = df_results[df_results['status'] == 'success']
                        if not success_df.empty:
                            for _, row in success_df.iterrows():
                                with st.expander(f"📄 {row['filename']}"):
                                    st.text_area(
                                        "Texte extrait:",
                                        value=row['text'],
                                        height=200,
                                        key=f"text_{row['filename']}"
                                    )
                        else:
                            st.info("Aucun fichier traité avec succès")
                    
                    with tab3:
                        error_df = df_results[df_results['status'] == 'error']
                        if not error_df.empty:
                            for _, row in error_df.iterrows():
                                st.error(f"**{row['filename']}**: {row['error']}")
                        else:
                            st.success("Aucune erreur!")
                    
                    # Téléchargement des résultats
                    if success_count > 0:
                        st.header("💾 Téléchargement")
                        
                        zip_data = processor.create_results_zip(results)
                        
                        st.download_button(
                            label="📦 Télécharger tous les résultats (ZIP)",
                            data=zip_data,
                            file_name=f"ocr_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                            mime="application/zip",
                            type="primary",
                            use_container_width=True
                        )
                        
                        st.info("""
                        **Le fichier ZIP contient:**
                        - `ocr_summary.csv`: Résumé au format CSV
                        - `extracted_texts/`: Dossier avec les textes extraits (.txt)
                        - `ocr_results.json`: Données complètes au format JSON
                        """)
        
        except Exception as e:
            st.error(f"Erreur lors de l'initialisation: {e}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Powered by <strong>Mistral AI</strong> | Développé avec ❤️ en Python & Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
