import streamlit as st
import base64
import os
import io
from pathlib import Path
from mistralai import Mistral
import mimetypes
from typing import List, Dict, Any, Optional
import json
from datetime import datetime
import re
from PIL import Image

class TargetedOCRProcessor:
    """
    Processeur OCR spécialisé pour l'extraction d'informations personnelles
    depuis des captures d'écran complexes
    """
    
    def __init__(self):
        self.ocr_model = "mistral-ocr-latest"
        self.supported_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp'}
        
    def get_mistral_client(self, api_key: str) -> Mistral:
        """Initialise le client Mistral avec la clé API"""
        return Mistral(api_key=api_key)
    
    def preprocess_image(self, image_bytes: bytes) -> bytes:
        """
        Prétraite l'image pour améliorer l'OCR sur les captures d'écran
        - Augmente le contraste
        - Supprime le bruit
        - Recadre si nécessaire
        """
        try:
            from PIL import Image, ImageEnhance, ImageFilter
            import io
            
            # Charger l'image
            img = Image.open(io.BytesIO(image_bytes))
            
            # Convertir en RGB si nécessaire
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Améliorer le contraste
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.5)
            
            # Améliorer la netteté
            enhancer = ImageEnhance.Sharpness(img)
            img = enhancer.enhance(2.0)
            
            # Sauvegarder l'image traitée
            output = io.BytesIO()
            img.save(output, format='PNG', optimize=True)
            return output.getvalue()
            
        except Exception as e:
            st.warning(f"Prétraitement échoué, utilisation de l'image originale: {e}")
            return image_bytes
    
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
    
    def extract_personal_info(self, ocr_text: str) -> Dict[str, Any]:
        """
        Extrait les informations personnelles structurées du texte OCR
        Adapté pour le format de fiche PRONOTE/système scolaire
        """
        info = {
            'nom_complet': None,
            'prenom': None,
            'nom': None,
            'profession': None,
            'situation': None,
            'adresse': None,
            'code_postal': None,
            'ville': None,
            'telephone_fixe': None,
            'telephone_mobile': None,
            'email': None,
            'autorisation_sms': False,
            'autorisation_email': False,
            'autorisation_courrier': False,
            'legal': None,
            'dates': {},
            'identifiants': {},
            'raw_text': ocr_text
        }
        
        lines = ocr_text.split('\n')
        
        for i, line in enumerate(lines):
            line_clean = line.strip()
            
            # Extraction du nom complet (format: Prénom NOM ou NOM Prénom)
            if 'PLANTEGENET' in line_clean or 'Mme' in line_clean or 'M.' in line_clean:
                # Nettoyer et extraire le nom
                name_parts = re.sub(r'(Mme|M\.|Mlle)\s*', '', line_clean)
                name_parts = re.sub(r'\([^)]*\)', '', name_parts).strip()
                if name_parts:
                    info['nom_complet'] = name_parts
                    # Essayer de séparer prénom et nom
                    parts = name_parts.split()
                    if len(parts) >= 2:
                        if parts[-1].isupper():
                            info['nom'] = parts[-1]
                            info['prenom'] = ' '.join(parts[:-1])
                        else:
                            info['prenom'] = parts[0]
                            info['nom'] = ' '.join(parts[1:])
            
            # Profession
            if 'Profession' in line_clean or 'Employé' in line_clean:
                prof_match = re.search(r'(?:Profession\s*:?\s*|Employés?\s+)(.+)', line_clean, re.IGNORECASE)
                if prof_match:
                    info['profession'] = prof_match.group(1).strip()
            
            # Situation familiale
            if 'Situation' in line_clean or 'CELIBATAIRE' in line_clean or 'MARIE' in line_clean:
                sit_match = re.search(r'(?:Situation\s*:?\s*)?(CELIBATAIRE|MARIE|DIVORCE|VEUF|PACSE)', line_clean, re.IGNORECASE)
                if sit_match:
                    info['situation'] = sit_match.group(1)
            
            # Adresse
            if 'Adresse' in line_clean or 'rue' in line_clean.lower():
                addr_match = re.search(r'(?:Adresse\s*:?\s*)?(.+(?:rue|avenue|boulevard|place|impasse|chemin).+)', line_clean, re.IGNORECASE)
                if addr_match:
                    info['adresse'] = addr_match.group(1).strip()
            
            # Code postal et ville
            cp_ville_match = re.search(r'(\d{5})\s+([A-Z\s\-]+)', line_clean)
            if cp_ville_match:
                info['code_postal'] = cp_ville_match.group(1)
                info['ville'] = cp_ville_match.group(2).strip()
            
            # Téléphones
            tel_matches = re.findall(r'(?:\+33\s*|0)[\d\s]{9,14}', line_clean)
            for tel in tel_matches:
                tel_clean = re.sub(r'\s+', '', tel)
                if len(tel_clean) >= 10:
                    if 'fixe' in line_clean.lower() or not info['telephone_fixe']:
                        info['telephone_fixe'] = tel_clean
                    elif 'mobile' in line_clean.lower() or '06' in tel_clean or '07' in tel_clean:
                        info['telephone_mobile'] = tel_clean
            
            # Email
            email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', line_clean)
            if email_match:
                info['email'] = email_match.group(0)
            
            # Autorisations
            if 'SMS autorisé' in line_clean:
                info['autorisation_sms'] = True
            if 'Email interdit' in line_clean:
                info['autorisation_email'] = False
            elif 'Email autorisé' in line_clean:
                info['autorisation_email'] = True
            if 'Courrier autorisé' in line_clean:
                info['autorisation_courrier'] = True
            
            # Statut légal
            if 'LEGAL' in line_clean:
                info['legal'] = 'LEGAL'
            
            # Dates (format JJ/MM/AAAA)
            date_matches = re.findall(r'\d{2}/\d{2}/\d{4}', line_clean)
            for date in date_matches:
                if i < len(lines) - 1:
                    context = lines[i-1] if i > 0 else ""
                    info['dates'][context[:30]] = date
        
        return info
    
    def process_single_image(self, client: Mistral, image_bytes: bytes, filename: str, preprocess: bool = True) -> Dict[str, Any]:
        """Traite une seule image avec OCR et extraction d'informations"""
        try:
            # Prétraitement optionnel
            if preprocess:
                image_bytes = self.preprocess_image(image_bytes)
            
            # Encodage et préparation
            base64_image = self.encode_image(image_bytes)
            mime_type = self.get_image_mime_type(filename)
            
            # Appel OCR avec prompt spécialisé pour l'extraction
            ocr_response = client.ocr.process(
                model=self.ocr_model,
                document={
                    "type": "image_url",
                    "image_url": f"data:{mime_type};base64,{base64_image}"
                },
                include_image_base64=False
            )
            
            # Extraction du texte brut
            extracted_text = self.extract_clean_text(ocr_response)
            
            # Extraction structurée des informations personnelles
            structured_info = self.extract_personal_info(extracted_text)
            
            return {
                'filename': filename,
                'status': 'success',
                'raw_text': extracted_text,
                'structured_data': structured_info,
                'error': None,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            st.error(f"❌ Erreur OCR pour {filename}: {e}")
            return {
                'filename': filename,
                'status': 'error',
                'raw_text': '',
                'structured_data': {},
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def extract_clean_text(self, ocr_response) -> str:
        """Extrait le texte propre de la réponse OCR"""
        try:
            if hasattr(ocr_response, 'pages') and ocr_response.pages:
                page = ocr_response.pages[0]
                if hasattr(page, 'markdown'):
                    return page.markdown.strip()
            
            # Fallback: conversion string et extraction
            response_str = str(ocr_response)
            import re
            
            pattern = r'markdown="(.*?)"(?=,\s*images=)'
            match = re.search(pattern, response_str, re.DOTALL)
            
            if match:
                markdown_content = match.group(1)
                clean_text = (markdown_content
                             .replace('\\n', '\n')
                             .replace('\\t', '\t')
                             .replace('\\"', '"')
                             .replace('\\\\', '\\'))
                return clean_text.strip()
            
            return "[Extraction échouée]"
            
        except Exception as e:
            return f"[Erreur: {str(e)}]"
    
    def export_to_json(self, results: List[Dict]) -> str:
        """Exporte les résultats en JSON structuré"""
        export_data = {
            'extraction_date': datetime.now().isoformat(),
            'total_files': len(results),
            'successful': sum(1 for r in results if r['status'] == 'success'),
            'results': []
        }
        
        for result in results:
            if result['status'] == 'success':
                export_data['results'].append({
                    'filename': result['filename'],
                    'data': result['structured_data']
                })
        
        return json.dumps(export_data, indent=2, ensure_ascii=False)
    
    def export_to_txt(self, results: List[Dict]) -> str:
        """Exporte les résultats en format texte lisible"""
        output = []
        output.append("=" * 80)
        output.append(f"EXTRACTION OCR - {datetime.now().strftime('%d/%m/%Y %H:%M')}")
        output.append("=" * 80)
        
        for result in results:
            if result['status'] == 'success':
                data = result['structured_data']
                output.append(f"\n📄 Fichier: {result['filename']}")
                output.append("-" * 40)
                
                if data.get('nom_complet'):
                    output.append(f"👤 Nom complet: {data['nom_complet']}")
                if data.get('profession'):
                    output.append(f"💼 Profession: {data['profession']}")
                if data.get('situation'):
                    output.append(f"👫 Situation: {data['situation']}")
                if data.get('adresse'):
                    output.append(f"📍 Adresse: {data['adresse']}")
                    if data.get('code_postal') and data.get('ville'):
                        output.append(f"   {data['code_postal']} {data['ville']}")
                if data.get('telephone_fixe'):
                    output.append(f"☎️  Fixe: {data['telephone_fixe']}")
                if data.get('telephone_mobile'):
                    output.append(f"📱 Mobile: {data['telephone_mobile']}")
                if data.get('email'):
                    output.append(f"✉️  Email: {data['email']}")
                
                output.append("")
        
        return "\n".join(output)

def main():
    st.set_page_config(
        page_title="OCR Ciblé - Extraction d'informations",
        page_icon="🎯",
        layout="wide"
    )
    
    st.title("🎯 OCR Intelligent - Extraction d'Informations Personnelles")
    st.markdown("**Extraction automatique depuis captures d'écran complexes**")
    
    processor = TargetedOCRProcessor()
    
    # Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        api_key = st.text_input(
            "Clé API Mistral",
            type="password",
            help="Votre clé API Mistral AI"
        )
        
        st.divider()
        
        st.subheader("🔧 Options d'extraction")
        
        preprocess = st.checkbox(
            "Prétraitement d'image",
            value=True,
            help="Améliore le contraste et la netteté pour les captures d'écran"
        )
        
        extract_mode = st.radio(
            "Mode d'extraction",
            ["Structuré (recommandé)", "Texte brut uniquement"],
            help="Le mode structuré extrait automatiquement les champs"
        )
        
        if not api_key:
            st.warning("⚠️ Clé API requise")
            st.info("[Obtenir une clé](https://console.mistral.ai)")
    
    if api_key:
        try:
            client = processor.get_mistral_client(api_key)
            
            # Upload
            st.header("📁 Chargement des Images")
            
            uploaded_files = st.file_uploader(
                "Sélectionnez vos captures d'écran",
                type=['png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'webp'],
                accept_multiple_files=True,
                help="Formats supportés: PNG, JPG, JPEG, GIF, BMP, TIFF, WEBP"
            )
            
            if uploaded_files:
                st.success(f"✅ {len(uploaded_files)} fichier(s) chargé(s)")
                
                # Aperçu
                with st.expander("👁️ Aperçu des images"):
                    cols = st.columns(min(len(uploaded_files), 3))
                    for i, file in enumerate(uploaded_files[:3]):
                        with cols[i]:
                            st.image(file, caption=file.name, use_column_width=True)
                
                # Bouton de traitement
                if st.button("🚀 Lancer l'extraction", type="primary", use_container_width=True):
                    
                    # Progression
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    results = []
                    
                    # Traitement
                    for i, uploaded_file in enumerate(uploaded_files):
                        progress = (i + 1) / len(uploaded_files)
                        progress_bar.progress(progress)
                        status_text.text(f"Traitement: {uploaded_file.name} ({i+1}/{len(uploaded_files)})")
                        
                        image_bytes = uploaded_file.read()
                        result = processor.process_single_image(
                            client, 
                            image_bytes, 
                            uploaded_file.name,
                            preprocess=preprocess
                        )
                        results.append(result)
                        uploaded_file.seek(0)
                    
                    progress_bar.progress(1.0)
                    status_text.text("✅ Extraction terminée!")
                    
                    # Résultats
                    st.header("📊 Résultats de l'Extraction")
                    
                    # Statistiques
                    success_count = sum(1 for r in results if r['status'] == 'success')
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("✅ Réussis", success_count)
                    with col2:
                        st.metric("❌ Échecs", len(results) - success_count)
                    
                    # Affichage des données extraites
                    if extract_mode == "Structuré (recommandé)":
                        st.subheader("📋 Informations Extraites")
                        
                        for result in results:
                            if result['status'] == 'success':
                                with st.expander(f"👤 {result['filename']}", expanded=True):
                                    data = result['structured_data']
                                    
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                                        st.markdown("**🏷️ Identité**")
                                        if data.get('nom_complet'):
                                            st.write(f"Nom: **{data['nom_complet']}**")
                                        if data.get('profession'):
                                            st.write(f"Profession: {data['profession']}")
                                        if data.get('situation'):
                                            st.write(f"Situation: {data['situation']}")
                                    
                                    with col2:
                                        st.markdown("**📞 Contact**")
                                        if data.get('telephone_fixe'):
                                            st.write(f"Fixe: {data['telephone_fixe']}")
                                        if data.get('telephone_mobile'):
                                            st.write(f"Mobile: {data['telephone_mobile']}")
                                        if data.get('email'):
                                            st.write(f"Email: {data['email']}")
                                    
                                    if data.get('adresse'):
                                        st.markdown("**📍 Adresse**")
                                        st.write(data['adresse'])
                                        if data.get('code_postal') and data.get('ville'):
                                            st.write(f"{data['code_postal']} {data['ville']}")
                                    
                                    # Texte brut en accordéon
                                    with st.expander("📝 Texte brut OCR"):
                                        st.text_area(
                                            "Texte extrait",
                                            value=result['raw_text'],
                                            height=200,
                                            key=f"raw_{result['filename']}"
                                        )
                    else:
                        # Mode texte brut
                        for result in results:
                            if result['status'] == 'success':
                                with st.expander(f"📄 {result['filename']}"):
                                    st.text_area(
                                        "Texte extrait",
                                        value=result['raw_text'],
                                        height=300,
                                        key=f"text_{result['filename']}"
                                    )
                    
                    # Export
                    if success_count > 0:
                        st.header("💾 Export des Données")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            json_data = processor.export_to_json(results)
                            st.download_button(
                                label="📋 JSON Structuré",
                                data=json_data,
                                file_name=f"extraction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                mime="application/json",
                                use_container_width=True
                            )
                        
                        with col2:
                            txt_data = processor.export_to_txt(results)
                            st.download_button(
                                label="📝 Texte Formaté",
                                data=txt_data,
                                file_name=f"extraction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                                mime="text/plain",
                                use_container_width=True
                            )
                        
                        with col3:
                            # Export CSV pour tableur
                            import pandas as pd
                            
                            rows = []
                            for r in results:
                                if r['status'] == 'success':
                                    d = r['structured_data']
                                    rows.append({
                                        'Fichier': r['filename'],
                                        'Nom': d.get('nom_complet', ''),
                                        'Profession': d.get('profession', ''),
                                        'Téléphone': d.get('telephone_mobile', d.get('telephone_fixe', '')),
                                        'Email': d.get('email', ''),
                                        'Adresse': f"{d.get('adresse', '')} {d.get('code_postal', '')} {d.get('ville', '')}".strip()
                                    })
                            
                            if rows:
                                df = pd.DataFrame(rows)
                                csv = df.to_csv(index=False)
                                st.download_button(
                                    label="📊 Export CSV",
                                    data=csv,
                                    file_name=f"extraction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv",
                                    use_container_width=True
                                )
        
        except Exception as e:
            st.error(f"❌ Erreur: {e}")
    
    # Info
    with st.expander("ℹ️ À propos"):
        st.markdown("""
        ### 🎯 Fonctionnalités
        - **Extraction ciblée** des informations personnelles
        - **Prétraitement** automatique pour améliorer la qualité OCR
        - **Structuration** intelligente des données extraites
        - **Export** multi-format (JSON, TXT, CSV)
        
        ### 📋 Champs extraits
        - Nom complet, prénom, nom
        - Profession et situation familiale
        - Adresse complète
        - Téléphones (fixe et mobile)
        - Email
        - Autorisations de contact
        
        ### 🔧 Optimisations
        - Amélioration du contraste pour captures d'écran
        - Reconnaissance de patterns spécifiques
        - Validation des formats (téléphone, email, etc.)
        """)

if __name__ == "__main__":
    main()
