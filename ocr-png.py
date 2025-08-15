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
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

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
    
    def create_pdf_report(self, results: List[Dict[str, Any]]) -> bytes:
        """Crée un rapport PDF des résultats OCR"""
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []
        
        # Styles personnalisés
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.darkblue
        )
        
        subtitle_style = ParagraphStyle(
            'CustomSubtitle',
            parent=styles['Heading2'],
            fontSize=16,
            spaceAfter=20,
            alignment=TA_LEFT,
            textColor=colors.darkblue
        )
        
        content_style = ParagraphStyle(
            'CustomContent',
            parent=styles['Normal'],
            fontSize=10,
            spaceAfter=12,
            alignment=TA_JUSTIFY,
            leftIndent=20
        )
        
        filename_style = ParagraphStyle(
            'FilenameStyle',
            parent=styles['Normal'],
            fontSize=12,
            spaceAfter=8,
            textColor=colors.darkgreen,
            fontName='Helvetica-Bold'
        )
        
        # Page de titre
        story.append(Paragraph("🤖 Rapport d'Extraction OCR", title_style))
        story.append(Spacer(1, 20))
        
        # Statistiques
        success_count = sum(1 for r in results if r['status'] == 'success')
        error_count = len(results) - success_count
        
        stats_data = [
            ['Métrique', 'Valeur'],
            ['📄 Total de fichiers', str(len(results))],
            ['✅ Extractions réussies', str(success_count)],
            ['❌ Erreurs', str(error_count)],
            ['📅 Date de traitement', datetime.now().strftime('%d/%m/%Y à %H:%M')],
            ['⚡ Taux de réussite', f'{(success_count/len(results)*100):.1f}%' if results else '0%']
        ]
        
        stats_table = Table(stats_data, colWidths=[3*inch, 2*inch])
        stats_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
        ]))
        
        story.append(stats_table)
        story.append(PageBreak())
        
        # Contenus extraits
        story.append(Paragraph("📋 Textes Extraits", subtitle_style))
        story.append(Spacer(1, 20))
        
        for i, result in enumerate(results):
            if result['status'] == 'success' and result['text'].strip():
                # Nom du fichier
                story.append(Paragraph(f"📄 {result['filename']}", filename_style))
                
                # Texte extrait (limité pour éviter les pages trop longues)
                text = result['text'].strip()
                if len(text) > 1000:
                    text = text[:1000] + "... [texte tronqué]"
                
                # Échapper les caractères spéciaux pour ReportLab
                text = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                
                story.append(Paragraph(text, content_style))
                story.append(Spacer(1, 15))
                
                # Saut de page tous les 3 documents pour éviter la surcharge
                if (i + 1) % 3 == 0 and i < len(results) - 1:
                    story.append(PageBreak())
        
        # Section des erreurs si il y en a
        error_results = [r for r in results if r['status'] == 'error']
        if error_results:
            story.append(PageBreak())
            story.append(Paragraph("⚠️ Erreurs de Traitement", subtitle_style))
            story.append(Spacer(1, 20))
            
            for result in error_results:
                story.append(Paragraph(f"❌ {result['filename']}", filename_style))
                error_text = result['error'].replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                story.append(Paragraph(f"Erreur: {error_text}", content_style))
                story.append(Spacer(1, 10))
        
        # Pied de page
        story.append(PageBreak())
        story.append(Spacer(1, 200))
        footer_style = ParagraphStyle(
            'Footer',
            parent=styles['Normal'],
            fontSize=10,
            alignment=TA_CENTER,
            textColor=colors.grey
        )
        story.append(Paragraph("Généré par OCR Streamlit App - Powered by Mistral AI", footer_style))
        
        doc.build(story)
        buffer.seek(0)
        return buffer.getvalue()
    
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
            
            # Rapport PDF
            pdf_data = self.create_pdf_report(results)
            zip_file.writestr('rapport_ocr.pdf', pdf_data)
        
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
                            # ZIP complet
                            zip_data = processor.create_results_zip(results)
                            st.download_button(
                                label="📦 Télécharger ZIP complet",
                                data=zip_data,
                                file_name=f"ocr_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                                mime="application/zip",
                                type="primary",
                                use_container_width=True
                            )
                        
                        with col2:
                            # PDF seul
                            pdf_data = processor.create_pdf_report(results)
                            st.download_button(
                                label="📄 Télécharger rapport PDF",
                                data=pdf_data,
                                file_name=f"rapport_ocr_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                                mime="application/pdf",
                                type="secondary",
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
                        - `rapport_ocr.pdf`: Rapport complet formaté
                        - `ocr_summary.csv`: Résumé au format CSV
                        - `extracted_texts/`: Dossier avec les textes extraits (.txt)
                        - `ocr_results.json`: Données complètes au format JSON
                        
                        **📄 Le rapport PDF inclut:**
                        - Statistiques détaillées du traitement
                        - Tous les textes extraits avec mise en page
                        - Rapport des erreurs (si applicable)
                        """)
                        
                        # Aperçu PDF dans l'interface
                        with st.expander("👁️ Aperçu du rapport PDF"):
                            st.markdown(f"""
                            **Titre:** 🤖 Rapport d'Extraction OCR  
                            **Date:** {datetime.now().strftime('%d/%m/%Y à %H:%M')}  
                            **Total fichiers:** {len(results)}  
                            **Succès:** {success_count}  
                            **Erreurs:** {error_count}  
                            **Taux de réussite:** {(success_count/len(results)*100):.1f}%
                            """)
                            
                            if success_count > 0:
                                st.write("**Aperçu des premières extractions:**")
                                for i, result in enumerate([r for r in results if r['status'] == 'success'][:3]):
                                    st.write(f"📄 **{result['filename']}**")
                                    preview_text = result['text'][:200]
                                    if len(result['text']) > 200:
                                        preview_text += "..."
                                    st.write(f"_{preview_text}_")
                                    st.write("---")
        
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
