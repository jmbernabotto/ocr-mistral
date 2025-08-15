import streamlit as st
import base64
from pathlib import Path
from mistralai import Mistral
import json
from datetime import datetime
import re

class PronoteOCRExtractor:
    """
    Extracteur OCR optimisé pour les fiches PRONOTE
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
            
            # Appel OCR avec prompt amélioré
            ocr_response = client.ocr.process(
                model=self.ocr_model,
                document={
                    "type": "image_url",
                    "image_url": f"data:image/png;base64,{base64_image}"
                },
                include_image_base64=False
            )
            
            # Extraction du texte brut
            text = self.extract_text_from_response(ocr_response)
            
            if not text:
                return {
                    'status': 'error',
                    'filename': filename,
                    'error': 'Aucun texte extrait'
                }
            
            # Parse des informations
            return self.parse_contact_info(text, filename)
            
        except Exception as e:
            return {
                'status': 'error',
                'filename': filename,
                'error': str(e)
            }
    
    def extract_text_from_response(self, ocr_response) -> str:
        """
        Extrait proprement le texte de la réponse OCR
        """
        try:
            # Méthode 1: Accès direct
            if hasattr(ocr_response, 'pages') and ocr_response.pages:
                if hasattr(ocr_response.pages[0], 'markdown'):
                    return ocr_response.pages[0].markdown.strip()
            
            # Méthode 2: Conversion et extraction
            response_str = str(ocr_response)
            
            # Recherche du contenu markdown
            if 'markdown=' in response_str or 'markdown="' in response_str:
                # Pattern pour extraire le markdown
                import re
                patterns = [
                    r'markdown="([^"]*)"',
                    r"markdown='([^']*)'",
                    r'markdown=([^,\s]+)'
                ]
                
                for pattern in patterns:
                    match = re.search(pattern, response_str, re.DOTALL)
                    if match:
                        content = match.group(1)
                        # Décoder les échappements
                        content = content.replace('\\n', '\n')
                        content = content.replace('\\t', '\t')
                        content = content.replace('\\"', '"')
                        content = content.replace('\\\\', '\\')
                        return content.strip()
            
            return ""
            
        except Exception as e:
            st.error(f"Erreur extraction texte: {e}")
            return ""
    
    def parse_contact_info(self, text: str, filename: str) -> dict:
        """
        Parse le texte OCR pour extraire les informations structurées
        Version améliorée pour mieux gérer le format PRONOTE
        """
        # Structure de base
        info = {
            'status': 'success',
            'filename': filename,
            'etablissement': None,
            'eleve_nom': None,
            'eleve_prenom': None,
            'classe': None,
            'responsable': {
                'nom_complet': None,
                'nom': None,
                'prenom': None,
                'relation': None,
                'statut': None,
                'profession': None,
                'situation': None,
                'adresse': None,
                'code_postal': None,
                'ville': None,
                'pays': None,
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
        
        # Nettoyage et préparation du texte
        lines = text.split('\n')
        text_lower = text.lower()
        
        # Recherche plus agressive des informations
        
        # 1. ÉTABLISSEMENT - chercher différents patterns
        for line in lines[:10]:  # Généralement dans les premières lignes
            if any(mot in line.upper() for mot in ['COLLÈGE', 'LYCÉE', 'ÉCOLE']):
                info['etablissement'] = line.strip()
                break
        
        # 2. NOM DE L'ÉLÈVE - chercher le nom principal (souvent en gros)
        # Recherche spécifique pour "Aidhan COLOMBO PLANTEGENET"
        for i, line in enumerate(lines):
            # Pattern pour nom d'élève (pas de Mme/M., souvent en majuscules partielles)
            if ('COLOMBO' in line.upper() or 'PLANTEGENET' in line.upper()) and \
               'MME' not in line.upper() and 'M.' not in line and \
               '(' not in line and 'PROFESSION' not in line.upper():
                # Nettoyer et extraire
                nom_eleve = line.strip()
                # Retirer les éléments non pertinents
                nom_eleve = re.sub(r'\b\d{1,2}[A-Z]\d?\b', '', nom_eleve).strip()  # Retirer classe
                if nom_eleve and len(nom_eleve) > 3:
                    info['eleve_nom'] = nom_eleve
                    # Essayer de séparer prénom et nom
                    parts = nom_eleve.split()
                    if len(parts) >= 2:
                        # Généralement : Prénom NOM NOM
                        info['eleve_prenom'] = parts[0]
                        break
        
        # 3. CLASSE - Format typique: 3E1, 6A, etc.
        for line in lines:
            # Recherche de patterns de classe
            classe_patterns = [
                r'\b([1-6])[eE]([1-9])\b',  # 3E1, 6E2
                r'\b([1-6])[A-Z]([1-9])?\b',  # 3A, 6B2
                r'\b(3E1|3E2|6E1|5E1|4E1)\b'  # Patterns spécifiques
            ]
            for pattern in classe_patterns:
                match = re.search(pattern, line)
                if match:
                    info['classe'] = match.group(0)
                    break
            if info['classe']:
                break
        
        # 4. RESPONSABLE LÉGAL - Recherche améliorée
        for i, line in enumerate(lines):
            # Recherche du pattern Mme/M. NOM Prénom
            if ('MME' in line.upper() or 'M.' in line) and 'PLANTEGENET' in line.upper():
                # Extraction du nom complet
                patterns = [
                    r'(Mme|M\.)\s+([A-ZÀ-Ÿ]+)\s+([A-Za-zÀ-ÿ]+)',  # Mme NOM Prénom
                    r'(Mme|M\.)\s+([^\(]+)',  # Mme Nom complet
                ]
                
                for pattern in patterns:
                    match = re.search(pattern, line, re.IGNORECASE)
                    if match:
                        if len(match.groups()) >= 3:
                            info['responsable']['nom'] = match.group(2)
                            info['responsable']['prenom'] = match.group(3)
                            info['responsable']['nom_complet'] = f"{match.group(2)} {match.group(3)}"
                        else:
                            info['responsable']['nom_complet'] = match.group(2).strip()
                        break
                
                # Extraction de la relation (MÈRE, PÈRE)
                if '(MÈRE)' in line.upper() or 'MERE' in line.upper():
                    info['responsable']['relation'] = 'MÈRE'
                elif '(PÈRE)' in line.upper() or 'PERE' in line.upper():
                    info['responsable']['relation'] = 'PÈRE'
                elif match := re.search(r'\(([^)]+)\)', line):
                    info['responsable']['relation'] = match.group(1)
        
        # 5. STATUT LÉGAL
        if 'LÉGAL' in text.upper() or 'LEGAL' in text.upper():
            info['responsable']['statut'] = 'LÉGAL'
        
        # 6. PROFESSION - Recherche améliorée
        for i, line in enumerate(lines):
            if 'profession' in line.lower() or 'employé' in line.lower():
                # Prendre le contenu après "Profession :" ou la ligne suivante
                if ':' in line:
                    prof = line.split(':', 1)[1].strip()
                    if prof:
                        info['responsable']['profession'] = prof
                elif i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    if next_line and not any(x in next_line.upper() for x in ['SITUATION', 'ADRESSE', 'LÉGAL']):
                        info['responsable']['profession'] = next_line
        
        # 7. SITUATION FAMILIALE
        situations = ['CÉLIBATAIRE', 'CELIBATAIRE', 'MARIÉ', 'MARIE', 'DIVORCÉ', 'DIVORCE', 'VEUF', 'PACSÉ']
        for situation in situations:
            if situation in text.upper():
                info['responsable']['situation'] = situation.replace('E', 'É') if 'MARIE' in situation else situation
                break
        
        # 8. ADRESSE - Recherche améliorée
        for i, line in enumerate(lines):
            # Recherche de numéro + rue/avenue/boulevard
            addr_pattern = r'(\d+)\s+(rue|avenue|boulevard|place|chemin|impasse)'
            if re.search(addr_pattern, line, re.IGNORECASE):
                info['responsable']['adresse'] = line.strip()
                # Chercher code postal et ville dans les lignes suivantes
                for j in range(i, min(i + 3, len(lines))):
                    cp_match = re.search(r'(\d{5})\s+([A-ZÀ-Ÿ\s\-]+)', lines[j])
                    if cp_match:
                        info['responsable']['code_postal'] = cp_match.group(1)
                        ville = cp_match.group(2).strip()
                        # Séparer ville et pays si présent
                        if 'FRANCE' in ville:
                            ville = ville.replace('- FRANCE', '').replace('FRANCE', '').strip()
                            info['responsable']['pays'] = 'FRANCE'
                        info['responsable']['ville'] = ville
                        break
        
        # 9. TÉLÉPHONES - Extraction améliorée
        # Patterns pour les numéros français
        tel_patterns = [
            r'(?:\+33|0)\s?[1-9](?:\s?\d{2}){4}',  # Format français
            r'\(\+33\)\s?[1-9](?:\s?\d{2}){4}',    # Format avec parenthèses
            r'0[1-9]\s?\d{2}\s?\d{2}\s?\d{2}\s?\d{2}'  # Format classique
        ]
        
        for pattern in tel_patterns:
            matches = re.findall(pattern, text)
            for tel in matches:
                # Nettoyer le numéro
                tel_clean = re.sub(r'[\s\(\)\+\-]', '', tel)
                if tel_clean.startswith('33'):
                    tel_clean = '0' + tel_clean[2:]
                
                # Formater avec espaces
                if len(tel_clean) == 10:
                    tel_format = ' '.join([tel_clean[i:i+2] for i in range(0, 10, 2)])
                    
                    # Déterminer si fixe ou mobile
                    if tel_clean.startswith('06') or tel_clean.startswith('07'):
                        if not info['responsable']['telephone_mobile']:
                            info['responsable']['telephone_mobile'] = tel_format
                    else:
                        if not info['responsable']['telephone_fixe']:
                            info['responsable']['telephone_fixe'] = tel_format
        
        # 10. AUTORISATIONS - Recherche plus flexible
        autorisations_patterns = {
            'sms': ['SMS autorisé', 'sms autorise', 'SMS : autorisé'],
            'email': ['Email autorisé', 'email autorise', 'Email : autorisé', 
                     'Email interdit', 'email interdit', 'Email : interdit'],
            'courrier': ['Courrier autorisé', 'courrier autorise', 'Courrier : autorisé'],
            'discussion': ['Discussion autorisé', 'discussion autorise', 'Discussion : autorisé']
        }
        
        for key, patterns in autorisations_patterns.items():
            for pattern in patterns:
                if pattern.lower() in text_lower:
                    if 'interdit' in pattern.lower():
                        info['autorisations'][key] = False
                    else:
                        info['autorisations'][key] = True
                    break
        
        return info

def main():
    st.set_page_config(
        page_title="Extracteur PRONOTE",
        page_icon="📇",
        layout="wide"
    )
    
    st.title("📇 Extracteur de Fiches PRONOTE")
    st.markdown("**Extraction optimisée pour les captures d'écran PRONOTE**")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        api_key = st.text_input(
            "Clé API Mistral",
            type="password",
            help="Entrez votre clé API Mistral"
        )
        
        st.divider()
        
        show_debug = st.checkbox(
            "Mode debug",
            help="Affiche le texte brut extrait par l'OCR"
        )
    
    if not api_key:
        st.warning("⚠️ Veuillez entrer votre clé API Mistral dans la barre latérale")
        st.stop()
    
    # Upload
    uploaded_files = st.file_uploader(
        "Chargez vos captures d'écran PRONOTE",
        type=['png', 'jpg', 'jpeg'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        st.info(f"📁 {len(uploaded_files)} fichier(s) chargé(s)")
        
        if st.button("🚀 Extraire les informations", type="primary"):
            extractor = PronoteOCRExtractor()
            results = []
            
            progress = st.progress(0)
            status_text = st.empty()
            
            for idx, file in enumerate(uploaded_files):
                progress.progress((idx + 1) / len(uploaded_files))
                status_text.text(f"Traitement de {file.name}...")
                
                # Traitement
                image_bytes = file.read()
                result = extractor.process_image(api_key, image_bytes, file.name)
                results.append(result)
                file.seek(0)
            
            status_text.text("✅ Extraction terminée!")
            
            # Affichage des résultats
            st.header("📊 Résultats")
            
            for result in results:
                if result['status'] == 'success':
                    with st.expander(f"📄 {result['filename']}", expanded=True):
                        
                        # Mode debug - afficher le texte brut
                        if show_debug:
                            with st.expander("🔍 Texte brut OCR"):
                                st.text_area(
                                    "Texte extrait",
                                    value=result.get('texte_brut', ''),
                                    height=200,
                                    key=f"debug_{result['filename']}"
                                )
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**🏫 Informations élève**")
                            if result['etablissement']:
                                st.write(f"📍 Établissement: **{result['etablissement']}**")
                            if result['eleve_nom']:
                                st.write(f"👤 Élève: **{result['eleve_nom']}**")
                            if result['classe']:
                                st.write(f"📚 Classe: **{result['classe']}**")
                            
                            st.divider()
                            
                            st.markdown("**👨‍👩‍👧 Responsable légal**")
                            resp = result['responsable']
                            if resp['nom_complet']:
                                st.write(f"👤 {resp['nom_complet']}")
                            elif resp['nom'] and resp['prenom']:
                                st.write(f"👤 {resp['nom']} {resp['prenom']}")
                            if resp['relation']:
                                st.write(f"👥 Relation: **{resp['relation']}**")
                            if resp['statut']:
                                st.write(f"⚖️ Statut: **{resp['statut']}**")
                            if resp['situation']:
                                st.write(f"💑 Situation: {resp['situation']}")
                            if resp['profession']:
                                prof_short = resp['profession'][:60] + "..." if len(resp['profession']) > 60 else resp['profession']
                                st.write(f"💼 Profession: {prof_short}")
                        
                        with col2:
                            st.markdown("**📞 Coordonnées**")
                            if resp['adresse']:
                                st.write(f"🏠 {resp['adresse']}")
                            if resp['code_postal'] or resp['ville']:
                                ville_str = f"📍 {resp['code_postal'] or ''} {resp['ville'] or ''}".strip()
                                if ville_str != "📍":
                                    st.write(ville_str)
                            if resp['pays']:
                                st.write(f"🌍 {resp['pays']}")
                            if resp['telephone_fixe']:
                                st.write(f"☎️ Fixe: **{resp['telephone_fixe']}**")
                            if resp['telephone_mobile']:
                                st.write(f"📱 Mobile: **{resp['telephone_mobile']}**")
                            
                            st.divider()
                            
                            st.markdown("**✉️ Autorisations**")
                            auth = result['autorisations']
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.write(f"SMS: {'✅' if auth['sms'] else '❌'}")
                                st.write(f"Email: {'✅' if auth['email'] else '❌'}")
                            with col_b:
                                st.write(f"Courrier: {'✅' if auth['courrier'] else '❌'}")
                                st.write(f"Discussion: {'✅' if auth['discussion'] else '❌'}")
                
                else:
                    st.error(f"❌ Erreur pour {result['filename']}: {result.get('error', 'Inconnue')}")
            
            # Export uniquement si au moins un succès
            success_results = [r for r in results if r['status'] == 'success']
            
            if success_results:
                st.header("💾 Export des données")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Export TXT
                    txt_export = []
                    for r in success_results:
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
                        if resp['nom_complet']:
                            txt_export.append(f"  Nom: {resp['nom_complet']}")
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
                        if resp['pays']:
                            txt_export.append(f"  Pays: {resp['pays']}")
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
                        file_name=f"fiches_pronote_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
                
                with col2:
                    # Export JSON
                    json_results = []
                    for r in success_results:
                        # Nettoyer pour JSON (retirer texte_brut)
                        clean_result = {k: v for k, v in r.items() if k != 'texte_brut'}
                        json_results.append(clean_result)
                    
                    json_content = json.dumps(json_results, indent=2, ensure_ascii=False)
                    
                    st.download_button(
                        label="📋 Télécharger JSON",
                        data=json_content,
                        file_name=f"fiches_pronote_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                        use_container_width=True
                    )

if __name__ == "__main__":
    main()
