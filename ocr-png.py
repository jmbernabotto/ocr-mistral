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
import fitz  # PyMuPDF

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
    
    def extract_clean_text(self, ocr_response) -> str:
        """Extrait le texte propre de la réponse OCR Mistral"""
        try:
            # Si c'est une réponse avec structure pages/markdown
            if isinstance(ocr_response, dict) and 'pages' in str(ocr_response):
                # Extraire le markdown du premier élément pages
                response_str = str(ocr_response)
                
                # Chercher le pattern markdown="..."
                import re
                markdown_match = re.search(r'markdown="([^"]*(?:\\.[^"]*)*)"', response_str)
                if markdown_match:
                    markdown_text = markdown_match.group(1)
                    # Décoder les échappements
                    clean_text = markdown_text.replace('\\n', '\n').replace('\\"', '"').replace('\\\\', '\\')
                    return clean_text.strip()
            
            # Méthodes d'extraction classiques
            if hasattr(ocr_response, 'text'):
                return ocr_response.text.strip()
            elif isinstance(ocr_response, dict):
                if 'text' in ocr_response:
                    return ocr_response['text'].strip()
                elif 'choices' in ocr_response and len(ocr_response['choices']) > 0:
                    choice = ocr_response['choices'][0]
                    if isinstance(choice, dict):
                        if 'message' in choice and 'content' in choice['message']:
                            return choice['message']['content'].strip()
                        elif 'text' in choice:
                            return choice['text'].strip()
                    return str(choice).strip()
                elif 'content' in ocr_response:
                    return ocr_response['content'].strip()
            
            return str(ocr_response).strip()
            
        except Exception as e:
            st.warning(f"Erreur extraction texte: {e}")
            return str(ocr_response)[:1000]  # Fallback avec limitation

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
                include_image_base64=False
            )
            
            # Extraction propre du texte
            extracted_text = self.extract_clean_text(ocr_response)
            
            # Debug simplifié
            st.write(f"✅ {filename}: {len(extracted_text)} caractères extraits")
            if len(extracted_text) > 0:
                preview = extracted_text[:100].replace('\n', ' ')
                st.write(f"📝 Aperçu: {preview}...")
            
            return {
                'filename': filename,
                'status': 'success',
                'text': extracted_text,
                'error': None,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            st.error(f"❌ Erreur OCR pour {filename}: {e}")
            return {
                'filename': filename,
                'status': 'error',
                'text': '',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def create_simple_pdf_fallback(self, filename: str, extracted_text: str) -> bytes:
        """Version de fallback pour créer un PDF simple sans polices spéciales"""
        try:
            doc = fitz.open()
            page = doc.new_page()
            
            # Texte simple sans spécification de police
            content = f"Fichier: {filename}\n\n{extracted_text if extracted_text else 'Aucun texte detecte'}"
            
            # Découper le texte en lignes pour éviter les débordements
            lines = content.split('\n')
            y_position = 50
            
            for line in lines:
                if y_position > 750:  # Nouvelle page si nécessaire
                    page = doc.new_page()
                    y_position = 50
                
                # Insérer ligne par ligne sans fontname
                page.insert_text((50, y_position), line[:80], fontsize=11)  # Limiter à 80 chars
                y_position += 15
            
            pdf_bytes = doc.tobytes()
            doc.close()
            return pdf_bytes
            
        except Exception as e:
            st.error(f"Erreur fallback PDF: {e}")
            return None
    
    def create_pdf_from_text(self, filename: str, extracted_text: str) -> bytes:
        """Crée un PDF contenant le texte extrait d'une image"""
        try:
            # Créer un nouveau document PDF
            doc = fitz.open()
            
            # Ajouter une page
            page = doc.new_page()
            
            # Définir les marges
            margin = 50
            page_width = page.rect.width
            page_height = page.rect.height
            
            # Titre avec le nom du fichier source
            title = f"Texte extrait de: {filename}"
            page.insert_text(
                (margin, margin + 20),
                title,
                fontsize=14,
                fontname="Helvetica-Bold",
                color=(0, 0, 0.8)
            )
            
            # Ligne de séparation
            line_y = margin + 40
            page.draw_line(
                fitz.Point(margin, line_y),
                fitz.Point(page_width - margin, line_y),
                color=(0.7, 0.7, 0.7),
                width=1
            )
            
            # Préparer le texte
            if not extracted_text or not extracted_text.strip():
                extracted_text = "Aucun texte détecté dans cette image."
            
            # Diviser le texte en pages si nécessaire
            # Calculer la hauteur disponible pour le texte
            available_height = page_height - margin - 80  # 80 pour le titre et l'espace
            chars_per_page = int(available_height / 15) * 80  # Approximation: 80 chars par ligne, 15px par ligne
            
            text_chunks = []
            remaining_text = extracted_text
            
            while remaining_text:
                if len(remaining_text) <= chars_per_page:
                    text_chunks.append(remaining_text)
                    break
                else:
                    # Trouver un point de coupure logique (espace, retour à la ligne)
                    cut_point = chars_per_page
                    while cut_point > chars_per_page * 0.8 and remaining_text[cut_point] not in [' ', '\n', '.', ',']:
                        cut_point -= 1
                    
                    if cut_point <= chars_per_page * 0.8:
                        cut_point = chars_per_page
                    
                    text_chunks.append(remaining_text[:cut_point])
                    remaining_text = remaining_text[cut_point:].lstrip()
            
            # Insérer le premier chunk sur la première page
            if text_chunks:
                text_rect = fitz.Rect(margin, margin + 60, page_width - margin, page_height - margin)
                page.insert_textbox(
                    text_rect,
                    text_chunks[0],
                    fontsize=11,
                    fontname="Helvetica",
                    align=fitz.TEXT_ALIGN_LEFT,
                    color=(0, 0, 0)
                )
                
                # Créer des pages supplémentaires pour les autres chunks
                for chunk in text_chunks[1:]:
                    page = doc.new_page()
                    text_rect = fitz.Rect(margin, margin, page_width - margin, page_height - margin)
                    page.insert_textbox(
                        text_rect,
                        chunk,
                        fontsize=11,
                        fontname="Helvetica",
                        align=fitz.TEXT_ALIGN_LEFT,
                        color=(0, 0, 0)
                    )
            
            # Sauvegarder le PDF en mémoire
            pdf_bytes = doc.tobytes()
            doc.close()
            
            return pdf_bytes
            
        except Exception as e:
            st.warning(f"Erreur police standard pour {filename}: {e}. Utilisation du mode fallback...")
            # Essayer la version de fallback
            return self.create_simple_pdf_fallback(filename, extracted_text)
    
    def create_results_zip(self, results: List[Dict[str, Any]]) -> bytes:
        """Crée un fichier ZIP avec les PDFs et TXT générés"""
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Fichier CSV avec résumé
            df = pd.DataFrame(results)
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            zip_file.writestr('ocr_summary.csv', csv_buffer.getvalue())
            
            # Créer PDF et TXT pour chaque image traitée avec succès
            pdf_count = 0
            txt_count = 0
            
            for result in results:
                if result['status'] == 'success':
                    filename_stem = Path(result['filename']).stem
                    original_text = result['text'] if result['text'] else ""
                    
                    # Créer le PDF avec le texte original ou message par défaut
                    text_for_pdf = original_text if original_text.strip() else "Aucun texte détecté dans cette image."
                    pdf_data = self.create_pdf_from_text(result['filename'], text_for_pdf)
                    if pdf_data:
                        zip_file.writestr(f'pdfs/{filename_stem}.pdf', pdf_data)
                        pdf_count += 1
                    else:
                        st.warning(f"Impossible de créer le PDF pour {result['filename']}")
                    
                    # Créer le fichier TXT avec SEULEMENT le texte propre
                    try:
                        if original_text.strip():
                            # Seulement le texte extrait, rien d'autre
                            txt_content = original_text.strip()
                        else:
                            # Si vraiment aucun texte
                            txt_content = "Aucun texte détecté dans cette image."
                        
                        zip_file.writestr(f'txt/{filename_stem}.txt', txt_content)
                        txt_count += 1
                    except Exception as e:
                        st.warning(f"Impossible de créer le TXT pour {result['filename']}: {e}")
            
            # Log du nombre de fichiers créés
            if pdf_count > 0 or txt_count > 0:
                st.success(f"✅ {pdf_count} PDF(s) et {txt_count} TXT créé(s) avec succès!")
            
            # Fichier JSON avec tous les détails (pour debug si nécessaire)
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
                        
                        # Options de téléchargement
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # ZIP avec PDFs et TXT
                            zip_data = processor.create_results_zip(results)
                            st.download_button(
                                label="📦 Télécharger ZIP (PDFs + TXT)",
                                data=zip_data,
                                file_name=f"ocr_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                                mime="application/zip",
                                type="primary",
                                use_container_width=True
                            )
                        
                        with col2:
                            # Téléchargement PDF individuel pour le premier fichier réussi
                            first_success = next((r for r in results if r['status'] == 'success'), None)
                            if first_success:
                                pdf_data = processor.create_pdf_from_text(
                                    first_success['filename'], 
                                    first_success['text']
                                )
                                if pdf_data:
                                    filename_stem = Path(first_success['filename']).stem
                                    st.download_button(
                                        label=f"📄 PDF: {filename_stem}",
                                        data=pdf_data,
                                        file_name=f"{filename_stem}.pdf",
                                        mime="application/pdf",
                                        type="secondary",
                                        use_container_width=True
                                    )
                        
                        # Section pour télécharger des PDFs individuels
                        with st.expander("📄 Télécharger PDFs individuels"):
                            success_results = [r for r in results if r['status'] == 'success']
                            
                            # Organiser en colonnes de 3
                            cols_per_row = 3
                            for i in range(0, len(success_results), cols_per_row):
                                cols = st.columns(cols_per_row)
                                
                                for j, result in enumerate(success_results[i:i+cols_per_row]):
                                    with cols[j]:
                                        pdf_data = processor.create_pdf_from_text(
                                            result['filename'], 
                                            result['text']
                                        )
                                        if pdf_data:
                                            filename_stem = Path(result['filename']).stem
                                            st.download_button(
                                                label=f"📄 {filename_stem}",
                                                data=pdf_data,
                                                file_name=f"{filename_stem}.pdf",
                                                mime="application/pdf",
                                                key=f"pdf_{i}_{j}",
                                                use_container_width=True
                                            )
                        
                        # Options supplémentaires
                        with st.expander("📋 Autres formats"):
                            # CSV seul
                            csv_data = df_results.to_csv(index=False)
                            st.download_button(
                                label="📊 Télécharger CSV",
                                data=csv_data,
                                file_name=f"ocr_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                            
                            # JSON seul
                            json_data = json.dumps(results, indent=2, ensure_ascii=False)
                            st.download_button(
                                label="🔗 Télécharger JSON",
                                data=json_data,
                                file_name=f"ocr_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                mime="application/json"
                            )
                        
                        st.info("""
                        **📦 Le fichier ZIP contient:**
                        - `pdfs/`: Dossier avec un PDF par image traitée
                        - `txt/`: Dossier avec un fichier TXT par image traitée
                        - `ocr_summary.csv`: Résumé au format CSV
                        - `ocr_results.json`: Données complètes au format JSON
                        
                        **📄 Chaque PDF contient:**
                        - Le nom du fichier source en en-tête
                        - Le texte extrait de l'image formaté
                        - Pagination automatique si le texte est long
                        
                        **📝 Chaque fichier TXT contient:**
                        - Header avec nom du fichier source et date
                        - Le texte brut extrait de l'image
                        - Format simple pour traitement automatique
                        """)
                        
                        # Aperçu des fichiers générés
                        with st.expander("👁️ Aperçu des fichiers générés"):
                            success_results = [r for r in results if r['status'] == 'success']
                            
                            if success_results:
                                st.write(f"**{len(success_results)} PDFs et {len(success_results)} TXT seront générés:**")
                                
                                for i, result in enumerate(success_results[:5]):  # Afficher max 5
                                    filename_stem = Path(result['filename']).stem
                                    preview_text = result['text'][:200] if result['text'] else "Aucun texte détecté"
                                    if result['text'] and len(result['text']) > 200:
                                        preview_text += "..."
                                    
                                    st.write(f"📄 **{filename_stem}.pdf** | 📝 **{filename_stem}.txt**")
                                    st.write(f"Source: {result['filename']}")
                                    st.write(f"Aperçu: _{preview_text}_")
                                    st.write("---")
                                
                                if len(success_results) > 5:
                                    st.write(f"... et {len(success_results) - 5} autres paires de fichiers (PDF + TXT)")
                            else:
                                st.write("Aucun fichier à générer (aucune extraction réussie)")
        
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
