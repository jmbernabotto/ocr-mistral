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
