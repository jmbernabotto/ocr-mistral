import streamlit as st
import base64
from pathlib import Path
from mistralai import Mistral
import json
from datetime import datetime
import re

class SimpleOCRExtractor:
    """
    Extracteur OCR simple pour fiches de contact PRONOTE/Scolaire
    """
    
    def __init__(self):
        self.ocr_model = "mistral-ocr-latest"
    
    def process_image(self, api_key: str, image_bytes: bytes, filename: str) -> dict:
        """
        Traite une image et extrait les informations de contact
        """
        try:
            # Client Mistral
            client = Mistral(api_key=api_key)
            
            # Encodage base64
            base64_image = base64.b64encode(image_bytes).decode('utf-8')
            
            # Appel OCR
            ocr_response = client.ocr.process(
                model=self.ocr_model,
                document={
                    "type": "image_url",
                    "image_url": f"data:image/png;base64,{base64_image}"
                },
                include_image_base64=False
            )
            
            # Extraction du texte
            text = ""
            if hasattr(ocr_response, 'pages') and ocr_response.pages:
                text = ocr_response.pages[0].markdown.strip()
            
            # Parse des informations
            return self.parse_contact_info(text, filename)
            
        except Exception as e:
            return {
                'status': 'error',
                'filename': filename,
                'error': str(e)
            }
    
    def parse_contact_info(self, text: str, filename: str) -> dict:
        """
        Parse le texte OCR pour extraire les informations structurées
        """
        info = {
            'status': 'success',
            'filename': filename,
            'etablissement': None,
            'eleve_nom': None,
            'classe': None,
            'responsable': {
                'nom': None,
                'relation': None,
                'statut': None,
                'profession': None,
                'situation': None,
                'adresse': None,
                'code_postal': None,
                'ville': None,
                'telephone_fixe': None,
                'telephone_mobile': None
            },
            'autorisations': {
                'sms': False,
                'email': False,
                'courrier': False,
                'discussion': False
            },
            'texte_brut': text
        }
        
        lines = text.split('\n')
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Établissement (souvent en haut avec COLLÈGE, LYCÉE, etc.)
            if 'COLLÈGE' in line or 'LYCÉE' in line or 'ÉCOLE' in line:
                info['etablissement'] = line.strip()
            
            # Nom de l'élève (souvent en gros ou après l'établissement)
            if 'COLOMBO' in line or 'PLANTEGENET' in line:
                # Chercher le nom complet de l'élève (pas le responsable)
                if 'Mme' not in line and 'M.' not in line and '(' not in line:
                    info['eleve_nom'] = line.strip()
            
            # Classe (format 3E1, 6A, etc.)
            classe_match = re.search(r'\b\d[A-Z]\d?\b', line)
            if classe_match:
                info['classe'] = classe_match.group(0)
            
            # Responsable (format: Mme/M. NOM Prénom (RELATION))
            if ('Mme' in line or 'M.' in line) and 'PLANTEGENET' in line:
                # Extraire nom complet
                nom_match = re.search(r'(Mme|M\.)\s+([^(]+)', line)
                if nom_match:
                    info['responsable']['nom'] = nom_match.group(2).strip()
                
                # Extraire relation (MÈRE, PÈRE, etc.)
                relation_match = re.search(r'\(([^)]+)\)', line)
                if relation_match:
                    info['responsable']['relation'] = relation_match.group(1)
            
            # Statut légal
            if 'LÉGAL' in line or 'LEGAL' in line:
                info['responsable']['statut'] = 'LÉGAL'
            
            # Profession
            if 'Profession' in line or 'Employé' in line:
                # Prendre le reste de la ligne ou la ligne suivante
                if ':' in line:
                    info['responsable']['profession'] = line.split(':', 1)[1].strip()
                elif i < len(lines) - 1 and 'Profession' in line:
                    info['responsable']['profession'] = lines[i + 1].strip()
            
            # Situation familiale
            if 'CÉLIBATAIRE' in line or 'MARIÉ' in line or 'DIVORCÉ' in line:
                for situation in ['CÉLIBATAIRE', 'MARIÉ', 'DIVORCÉ', 'VEUF', 'PACSÉ']:
                    if situation in line:
                        info['responsable']['situation'] = situation
                        break
            
            # Adresse
            if 'rue' in line.lower() or 'avenue' in line.lower() or 'boulevard' in line.lower():
                # Extraire l'adresse
                addr_match = re.search(r'\d+\s+.*(rue|avenue|boulevard|place|chemin).*', line, re.IGNORECASE)
                if addr_match:
                    info['responsable']['adresse'] = addr_match.group(0).strip()
            
            # Code postal et ville
            cp_ville_match = re.search(r'(\d{5})\s+([A-Z][A-Z\s\-]+)', line)
            if cp_ville_match:
                info['responsable']['code_postal'] = cp_ville_match.group(1)
                ville = cp_ville_match.group(2).strip()
                # Retirer "FRANCE" si présent
                ville = ville.replace('- FRANCE', '').replace('FRANCE', '').strip()
                info['responsable']['ville'] = ville
            
            # Téléphones
            tel_matches = re.findall(r'(?:\+33\s*|0)[\d\s]{9,14}', line)
            for tel in tel_matches:
                tel_clean = re.sub(r'[\s\(\)\+]', '', tel)
                if tel_clean.startswith('33'):
                    tel_clean = '0' + tel_clean[2:]
                
                # Déterminer si fixe ou mobile
                if tel_clean.startswith('06') or tel_clean.startswith('07'):
                    info['responsable']['telephone_mobile'] = tel_clean
                else:
                    info['responsable']['telephone_fixe'] = tel_clean
            
            # Autorisations
            if 'SMS autorisé' in line:
                info['autorisations']['sms'] = True
            if 'Email autorisé' in line:
                info['autorisations']['email'] = True
            elif 'Email interdit' in line:
                info['autorisations']['email'] = False
            if 'Courrier autorisé' in line:
                info['autorisations']['courrier'] = True
            if 'Discussion autorisé' in line:
                info['autorisations']['discussion'] = True
        
        return info

def main():
    st.set_page_config(
        page_title="Extracteur Fiche Contact",
        page_icon="📇",
        layout="wide"
    )
    
    st.title("📇 Extracteur de Fiches Contact Scolaires")
    st.markdown("**Extraction simple et rapide des informations de contact**")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        api_key = st.text_input(
            "Clé API Mistral",
            type="password",
            help="Entrez votre clé API Mistral"
        )
        
        st.divider()
        
        format_export = st.radio(
            "Format d'export",
            ["Texte simple", "JSON", "Les deux"],
            index=2
        )
    
    if not api_key:
        st.warning("⚠️ Veuillez entrer votre clé API Mistral dans la barre latérale")
        st.stop()
    
    # Upload
    uploaded_files = st.file_uploader(
        "Chargez vos captures d'écran",
        type=['png', 'jpg', 'jpeg'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        st.info(f"📁 {len(uploaded_files)} fichier(s) chargé(s)")
        
        if st.button("🚀 Extraire les informations", type="primary"):
            extractor = SimpleOCRExtractor()
            results = []
            
            progress = st.progress(0)
            
            for idx, file in enumerate(uploaded_files):
                progress.progress((idx + 1) / len(uploaded_files))
                
                # Traitement
                image_bytes = file.read()
                result = extractor.process_image(api_key, image_bytes, file.name)
                results.append(result)
                file.seek(0)
            
            st.success("✅ Extraction terminée!")
            
            # Affichage des résultats
            st.header("📊 Résultats")
            
            for result in results:
                if result['status'] == 'success':
                    with st.expander(f"📄 {result['filename']}", expanded=True):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**🏫 Informations élève**")
                            if result['etablissement']:
                                st.write(f"Établissement: {result['etablissement']}")
                            if result['eleve_nom']:
                                st.write(f"Élève: **{result['eleve_nom']}**")
                            if result['classe']:
                                st.write(f"Classe: {result['classe']}")
                            
                            st.markdown("**👤 Responsable**")
                            resp = result['responsable']
                            if resp['nom']:
                                st.write(f"Nom: **{resp['nom']}**")
                            if resp['relation']:
                                st.write(f"Relation: {resp['relation']}")
                            if resp['statut']:
                                st.write(f"Statut: {resp['statut']}")
                            if resp['profession']:
                                st.write(f"Profession: {resp['profession'][:50]}...")
                            if resp['situation']:
                                st.write(f"Situation: {resp['situation']}")
                        
                        with col2:
                            st.markdown("**📍 Coordonnées**")
                            if resp['adresse']:
                                st.write(f"Adresse: {resp['adresse']}")
                            if resp['code_postal'] and resp['ville']:
                                st.write(f"{resp['code_postal']} {resp['ville']}")
                            if resp['telephone_fixe']:
                                st.write(f"Tél. fixe: {resp['telephone_fixe']}")
                            if resp['telephone_mobile']:
                                st.write(f"Tél. mobile: {resp['telephone_mobile']}")
                            
                            st.markdown("**✅ Autorisations**")
                            auth = result['autorisations']
                            st.write(f"SMS: {'✅' if auth['sms'] else '❌'}")
                            st.write(f"Email: {'✅' if auth['email'] else '❌'}")
                            st.write(f"Courrier: {'✅' if auth['courrier'] else '❌'}")
                            st.write(f"Discussion: {'✅' if auth['discussion'] else '❌'}")
                
                else:
                    st.error(f"❌ Erreur pour {result['filename']}: {result.get('error', 'Inconnue')}")
            
            # Export
            st.header("💾 Export")
            
            # Préparer les exports
            if format_export in ["Texte simple", "Les deux"]:
                txt_export = []
                for r in results:
                    if r['status'] == 'success':
                        txt_export.append(f"{'='*60}")
                        txt_export.append(f"FICHIER: {r['filename']}")
                        txt_export.append(f"{'='*60}")
                        if r['etablissement']:
                            txt_export.append(f"ÉTABLISSEMENT: {r['etablissement']}")
                        if r['eleve_nom']:
                            txt_export.append(f"ÉLÈVE: {r['eleve_nom']}")
                        if r['classe']:
                            txt_export.append(f"CLASSE: {r['classe']}")
                        
                        txt_export.append(f"\nRESPONSABLE LÉGAL:")
                        resp = r['responsable']
                        if resp['nom']:
                            txt_export.append(f"  Nom: {resp['nom']}")
                        if resp['relation']:
                            txt_export.append(f"  Relation: {resp['relation']}")
                        if resp['statut']:
                            txt_export.append(f"  Statut: {resp['statut']}")
                        if resp['profession']:
                            txt_export.append(f"  Profession: {resp['profession']}")
                        if resp['situation']:
                            txt_export.append(f"  Situation: {resp['situation']}")
                        if resp['adresse']:
                            txt_export.append(f"  Adresse: {resp['adresse']}")
                        if resp['code_postal'] and resp['ville']:
                            txt_export.append(f"  {resp['code_postal']} {resp['ville']}")
                        if resp['telephone_fixe']:
                            txt_export.append(f"  Tél. fixe: {resp['telephone_fixe']}")
                        if resp['telephone_mobile']:
                            txt_export.append(f"  Tél. mobile: {resp['telephone_mobile']}")
                        
                        txt_export.append(f"\nAUTORISATIONS:")
                        auth = r['autorisations']
                        txt_export.append(f"  SMS: {'OUI' if auth['sms'] else 'NON'}")
                        txt_export.append(f"  Email: {'OUI' if auth['email'] else 'NON'}")
                        txt_export.append(f"  Courrier: {'OUI' if auth['courrier'] else 'NON'}")
                        txt_export.append(f"  Discussion: {'OUI' if auth['discussion'] else 'NON'}")
                        txt_export.append("\n")
                
                txt_content = "\n".join(txt_export)
                
                st.download_button(
                    label="📝 Télécharger TXT",
                    data=txt_content,
                    file_name=f"fiches_contact_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
            
            if format_export in ["JSON", "Les deux"]:
                # Nettoyer pour JSON (retirer texte_brut pour alléger)
                json_results = []
                for r in results:
                    if r['status'] == 'success':
                        clean_result = {k: v for k, v in r.items() if k != 'texte_brut'}
                        json_results.append(clean_result)
                
                json_content = json.dumps(json_results, indent=2, ensure_ascii=False)
                
                st.download_button(
                    label="📋 Télécharger JSON",
                    data=json_content,
                    file_name=f"fiches_contact_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )

if __name__ == "__main__":
    main()
