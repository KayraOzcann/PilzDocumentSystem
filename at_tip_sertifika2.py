#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AT Tip İncelemesi Belgesi Analyzer (EC Type-Examination Certificate Analyzer)
Türkçe & İngilizce karma desenlerle güçlendirilmiş hâli.
2006/42/EC Ek IX uyumlu.
"""

import PyPDF2
import pytesseract
from pdf2image import convert_from_path
import re
import logging
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from datetime import datetime
import os
from langdetect import detect

# ---------- OCR & Poppler ----------
#pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
#poppler_path = r"C:\Users\nuvo_teknik_2\Desktop\poppler-24.08.0\Library\bin"
#os.environ["PATH"] += os.pathsep + poppler_path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ---------- Data Class ----------
@dataclass
class ATTipIncelemeResult:
    criteria_name: str
    found: bool
    content: str
    score: int
    max_score: int
    is_critical: bool
    details: Dict[str, Any]

# ---------- Analyzer ----------
class ATTipIncelemeAnalyzer:
    def __init__(self):
        logging.info("AT Type-Examination Certificate analysis system starting...")
        self.criteria_weights = {
            "Onaylanmış Kuruluş Bilgileri": 20,
            "Başvuru Sahibi/İmalatçı Bilgileri": 20,
            "Makine Tanımı": 15,
            "İncelenen Tip Tanımı": 10,
            "Uygulanan Hükümler": 15,
            "Değerlendirme Sonucu": 10,
            "Belge Geçerlilik Bilgileri": 10
        }

        # ---------- TÜRKÇE & İNGİLİZCE KARMA DESENLER ----------
        self.criteria_details = {
            "Onaylanmış Kuruluş Bilgileri": {
                "kurulusun_adi": {
                    "pattern": r"(?:notified\s+body|onaylanmış\s+kuruluş|onaylı\s+kuruluş|nb|bureau\s+veritas|tuv|sgs|dekra|intertek|bsi|lloyd's\s+register|dnv|kiwa|icim|csi|mts)[\s\w]*([A-Za-zÇŞİĞÜÖıçşığüö\s\.\-&]{5,80})|([A-Za-zÇŞİĞÜÖıçşığüö\s&\.]{5,50})\s*(?:ltd|gmbh|inc|corp|ag|certification|testing|inspection|prüfung|notified\s+body)",
                    "weight": 7, "critical": True, "description": "Onaylanmış kuruluşun adı"
                },
                "kurulusun_adresi": {
                    "pattern": r"(?:address|adres|adresi|konumu|yeri|sede|adresse)\s*[:\-]?\s*([A-Za-zÇŞİĞÜÖıçşığüö0-9\s\.\-/,&]{20,150})|(?:street|road|avenue|str\.|strasse|calle|via|cadde|sokak)[\s\w]*([A-Za-zÇŞİĞÜÖıçşığüö0-9\s\.\-/,&]{15,100})|(?:[0-9]{1,5}\s+[A-Za-zÇŞİĞÜÖıçşığüö\s]{5,50}(?:street|road|avenue|str\.|strasse|cadde|sokak))|(?:D-[0-9]{5}\s+[A-Za-zÇŞİĞÜÖıçşığüö]+)",
                    "weight": 6, "critical": True, "description": "Onaylanmış kuruluşun adresi"
                },
                "kimlik_numarasi": {
                    "pattern": r"(?:notified\s+body|nb|identification|kimlik|id|number|numarası|no|nummer|número)\s*[:\-]?\s*([0-9]{4})|(?:nb\s*[0-9]{4})|([0-9]{4})(?:\s*(?:notified|onaylanmış))",
                    "weight": 7, "critical": True, "description": "Onaylanmış kuruluş kimlik numarası (4 haneli)"
                }
            },
            "Başvuru Sahibi/İmalatçı Bilgileri": {
                "imalatci_adi": {
                    "pattern": r"(?:manufacturer|imalatçı|imalatci|fabrika|üretici|fabricant|hersteller|applicant|başvuru\s+sahibi|müracaatçı|company|şirket|firma|üretim\s+yeri)[\s:]*([A-Za-zÇŞİĞÜÖıçşığüö\s\.\-&]{5,100})|(?:we\s+hereby\s+certify\s+that\s+)([A-Za-zÇŞİĞÜÖıçşığüö\s&\.]+)|(?:this\s+certificate\s+is\s+issued\s+to\s+)([A-Za-zÇŞİĞÜÖıçşığüö\s&\.]+)",
                    "weight": 10, "critical": True, "description": "İmalatçı veya yetkili temsilcinin adı"
                },
                "imalatci_adres": {
                    "pattern": r"(?:manufacturer\s+address|imalatçı\s+adres|imalatci\s+adres|adresse\s+du\s+fabricant|herstelleradresse|üretici\s+adresi)\s*[:\-]?\s*([A-Za-zÇŞİĞÜÖıçşığüö0-9\s\.\-/,&]{20,150})|(?:located\s+at|registered\s+at|address|adres|konum|yer)\s*[:\-]?\s*([A-Za-zÇŞİĞÜÖıçşığüö0-9\s\.\-/,&]{15,120})",
                    "weight": 10, "critical": True, "description": "İmalatçı veya yetkili temsilcinin tam adresi"
                }
            },
            "Makine Tanımı": {
                "ticari_ad_tip": {
                    "pattern": r"(?:trade\s+name|ticari\s+ad|ticari\s+isim|commercial\s+name|product\s+name|denomination|type|tip|model|bezeichnung|ürün\s+adı|makine\s+adı)\s*[:\-]?\s*([A-Za-zÇŞİĞÜÖıçşığüö0-9\s\.\-/]{3,80})|(?:machine\s+type|makine\s+tipi|makine\s+modeli)\s*[:\-]?\s*([A-Za-zÇŞİĞÜÖıçşığüö0-9\s\.\-/]{3,80})",
                    "weight": 8, "critical": True, "description": "Makinenin ticari adı, tipi, modeli"
                },
                "seri_numarasi": {
                    "pattern": r"(?:serial\s+number|seri\s+numarası|seri\s+no|s/n|sn|série|seriennummer|sıra\s+no|üretim\s+no)\s*[:\-]?\s*([A-Za-zÇŞİĞÜÖıçşığüö0-9\-/\.]{2,25})|(?:serial)\s*[:\-]?\s*([A-Za-zÇŞİĞÜÖıçşığüö0-9\-/\.]{2,25})",
                    "weight": 4, "critical": True, "description": "Seri numarası veya tanımlamayı sağlayan bilgiler"
                },
                "varyantlar": {
                    "pattern": r"(?:variant|varyant|version|versions|sürüm|model\s+variants|configuration|konfigürasyon|seçenekler)\s*[:\-]?\s*([A-Za-zÇŞİĞÜÖıçşığüö0-9\s\.\-/,]{5,100})",
                    "weight": 3, "critical": False, "description": "Varyantlar veya versiyonlar (varsa)"
                }
            },
            "İncelenen Tip Tanımı": {
                "detayli_tanim": {
                    "pattern": r"(?:detailed\s+description|ayrıntılı\s+tanım|detaylı\s+açıklama|description\s+of\s+the\s+machine|machine\s+description|technical\s+description|makine\s+açıklaması|teknik\s+tanım)\s*[:\-]?\s*([A-Za-zÇŞİĞÜÖıçşığüö0-9\s\.\-/,&]{10,200})",
                    "weight": 4, "critical": True, "description": "İncelenen tipin ayrıntılı tanımı"
                },
                "teknik_dosya_atif": {
                    "pattern": r"(?:technical\s+file|teknik\s+dosya|teknik\s+evrak|technical\s+documentation|dossier\s+technique|technische\s+unterlage|documentation|reference|belge|dokümantasyon)\s*[:\-]?\s*([A-Za-zÇŞİĞÜÖıçşığüö0-9\s\.\-/]{5,50})|(?:according\s+to|göre|in\s+accordance\s+with)\s+(?:technical\s+file|documentation|teknik\s+dosya)",
                    "weight": 3, "critical": True, "description": "İlgili teknik dosyaya atıf"
                },
                "resim_plan_sema": {
                    "pattern": r"(?:drawing|plan|schema|şema|resim|picture|figure|şekil|diagram|blueprint|çizim|photos|fotoğraf|plan|poz|vaziyet\s+planı|montaj\s+resmi)\s*[:\-]?\s*([A-Za-zÇŞİĞÜÖıçşığüö0-9\s\.\-/]{3,50})|(?:as\s+shown\s+in|gösterildiği\s+gibi|according\s+to\s+drawing|çizime\s+göre)",
                    "weight": 3, "critical": False, "description": "Resim, plan, şema, parça listeleri"
                }
            },
            "Uygulanan Hükümler": {
                "direktif_atif": {
                    "pattern": r"(?:2006/42/EC|2006\/42\/EC|machinery\s+directive|makine\s+direktifi|directive\s+2006/42|machine\s+safety\s+directive|makine\s+emniyet\s+direktifi)",
                    "weight": 8, "critical": True, "description": "2006/42/EC direktif maddelerine atıf"
                },
                "uyumlastirilmis_standartlar": {
                    "pattern": r"(?:EN\s*ISO\s*[0-9]{3,5}[\-:]*[0-9]*[\-:]*[0-9]*|EN\s*[0-9]{3,5}[\-:]*[0-9]*[\-:]*[0-9]*|ISO\s*[0-9]{3,5}[\-:]*[0-9]*[\-:]*[0-9]*|IEC\s*[0-9]{3,5}[\-:]*[0-9]*[\-:]*[0-9]*)",
                    "weight": 5, "critical": True, "description": "Uyumlaştırılmış standartlar (EN ISO, EN IEC vb.)"
                },
                "esdeger_cozumler": {
                    "pattern": r"(?:equivalent\s+solution|eşdeğer\s+çözüm|alternative\s+solution|other\s+technical\s+solution|diğer\s+teknik\s+çözüm|non-harmonised|harmonize\s+olmayan|alternatif\s+çözüm)",
                    "weight": 2, "critical": False, "description": "Eşdeğer çözümlerin açıklaması (varsa)"
                }
            },
            "Değerlendirme Sonucu": {
                "uygunluk_ifadesi": {
                    "pattern": r"(?:complies\s+with|uygun|conform|conforms\s+to|in\s+compliance|meets\s+the\s+requirements|requirements\s+of|satisfies|karşılar|uygunluğu|conformity|compliance|uygundur|uygun\s+olduğu|uygunluk\s+ifadesi)",
                    "weight": 6, "critical": True, "description": "Direktif hükümlerine uygunluk ifadesi"
                },
                "test_muayene_ozet": {
                    "pattern": r"(?:test|muayene|examination\s+carried\s+out|inspection|assessment|değerlendirme|inceleme|kontrolü|yapılan\s+testler|performed\s+tests|evaluated|examined|test\s+sonuçları|muayene\s+raporu)",
                    "weight": 4, "critical": True, "description": "Yapılan testler/muayeneler/hesaplamaların özeti"
                }
            },
            "Belge Geçerlilik Bilgileri": {
                "duzenleme_tarihi": {
                    "pattern": r"(?:date|tarih|datum|fecha|düzenlenme\s+tarihi|issue\s+date)\s*[:\-]?\s*([0-9]{1,2}[\.\/\-][0-9]{1,2}[\.\/\-][0-9]{2,4})|([0-9]{1,2}\s+[A-Za-zÇŞİĞÜÖıçşığüö]{3,9}\s+[0-9]{4})|([0-9]{4}[\.\/\-][0-9]{1,2}[\.\/\-][0-9]{1,2})",
                    "weight": 3, "critical": True, "description": "Belgenin düzenlenme tarihi"
                },
                "belge_numarasi": {
                    "pattern": r"(?:certificate\s+number|belge\s+numarası|sertifika\s+no|cert\.\s*no\.?|number|nummer|número|ref|reference)\s*[:\-]?\s*([A-Za-zÇŞİĞÜÖıçşığüö0-9\-/\.]{5,30})|(?:cert\.\s*no\.?)\s*[:\-]?\s*([A-Za-zÇŞİĞÜÖıçşığüö0-9\-/\.]{5,30})",
                    "weight": 4, "critical": True, "description": "Belge numarası"
                },
                "gecerlilik_suresi": {
                    "pattern": r"(?:valid\s+until|geçerli|validity|expires|expiry\s+date|son\s+geçerlilik|until|bis|geçerlilik\s+süresi)\s*[:\-]?\s*([0-9]{1,2}[\.\/\-][0-9]{1,2}[\.\/\-][0-9]{2,4})|(?:this\s+certificate\s+is\s+valid)|(?:remains\s+valid)",
                    "weight": 2, "critical": False, "description": "Geçerlilik süresi (varsa)"
                },
                "yetkili_imza": {
                    "pattern": r"(?:signed\s+by|imzalayan|signature|imza|authorized\s+by|yetkili\s+temsilci|responsible\s+person|sorumlu\s+kişi|signatory|signed\s+for|on\s+behalf|imza\s+yetkilisi)",
                    "weight": 1, "critical": True, "description": "Yetkili temsilcinin imzası"
                }
            }
        }

    # ---------- Metin Çıkarım ----------
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    text += page.extract_text() or ""
            logging.info(f"PyPDF2 extracted {len(text)} characters")
            if len(text.strip()) < 100:
                logging.info("Insufficient text with PyPDF2, trying OCR...")
                pages = convert_from_path(pdf_path, dpi=200)
                ocr_text = ""
                for i, page in enumerate(pages, 1):
                    try:
                        page_text = pytesseract.image_to_string(page, lang='tur+eng+deu+fra+spa')
                        ocr_text += page_text + "\n"
                        logging.info(f"OCR extracted {len(page_text)} characters from page {i}")
                    except Exception as e:
                        logging.warning(f"OCR failed for page {i}: {e}")
                        continue
                if len(ocr_text.strip()) > len(text.strip()):
                    text = ocr_text
                    logging.info(f"OCR total text length: {len(text)}")
        except Exception as e:
            logging.error(f"Error extracting text: {e}")
            raise
        return text

    # ---------- Dil Tespiti ----------
    def detect_language(self, text: str) -> str:
        try:
            return detect(text) if len(text.strip()) >= 50 else "en"
        except:
            return "en"

    # ---------- Kriter Analizi ----------
    def analyze_criteria(self, text: str, category: str) -> Dict[str, ATTipIncelemeResult]:
        results = {}
        criteria = self.criteria_details.get(category, {})
        for criterion_name, criterion_data in criteria.items():
            pattern = criterion_data["pattern"]
            weight = criterion_data["weight"]
            is_critical = criterion_data["critical"]
            description = criterion_data["description"]
            matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
            if matches:
                clean_matches = []
                for match in matches:
                    if isinstance(match, tuple):
                        clean_match = next((m for m in match if m.strip()), "")
                    else:
                        clean_match = str(match)
                    if clean_match.strip():
                        clean_matches.append(clean_match.strip())
                if clean_matches:
                    content = f"Bulundu: {clean_matches[0][:60]}..."
                    found = True
                    score = weight
                else:
                    content = "Eşleşme bulundu ama değer çıkarılamadı"
                    found = True
                    score = int(weight * 0.5)
            else:
                content = "Bulunamadı"
                found = False
                score = 0
            results[criterion_name] = ATTipIncelemeResult(
                criteria_name=criterion_name,
                found=found,
                content=content,
                score=score,
                max_score=weight,
                is_critical=is_critical,
                details={"description": description, "pattern_used": pattern,
                         "matches_count": len(matches) if matches else 0,
                         "raw_matches": matches[:3] if matches else []}
            )
        return results

    # ---------- Puanlama ----------
    def calculate_scores(self, analysis_results: Dict[str, Dict[str, ATTipIncelemeResult]]) -> Dict[str, Any]:
        category_scores = {}
        total_score = 0
        critical_missing = []
        for category, results in analysis_results.items():
            category_max = self.criteria_weights[category]
            category_earned = sum(result.score for result in results.values())
            category_possible = sum(result.max_score for result in results.values())
            for criterion_name, result in results.items():
                if result.is_critical and not result.found:
                    critical_missing.append(f"{category}: {result.details['description']}")
            if category_possible > 0:
                percentage = (category_earned / category_possible) * 100
                normalized_score = (percentage / 100) * category_max
            else:
                percentage = normalized_score = 0
            category_scores[category] = {
                "earned": category_earned,
                "possible": category_possible,
                "normalized": round(normalized_score, 2),
                "max_weight": category_max,
                "percentage": round(percentage, 2)
            }
            total_score += normalized_score
        return {
            "category_scores": category_scores,
            "total_score": round(total_score, 2),
            "percentage": round(total_score, 2),
            "critical_missing": critical_missing
        }

    # ---------- Özel Değer Çıkarım ----------
    def extract_specific_values(self, text: str) -> Dict[str, Any]:
        values = {
            "notified_body_name": "Bulunamadı",
            "notified_body_address": "Bulunamadı",
            "notified_body_id": "Bulunamadı",
            "manufacturer_name": "Bulunamadı",
            "manufacturer_address": "Bulunamadı",
            "machine_trade_name": "Bulunamadı",
            "machine_type": "Bulunamadı",
            "machine_model": "Bulunamadı",
            "serial_number": "Bulunamadı",
            "certificate_number": "Bulunamadı",
            "issue_date": "Bulunamadı",
            "validity_date": "Bulunamadı",
            "directive_reference": "Bulunamadı",
            "applied_standards": [],
            "authorized_person": "Bulunamadı"
        }

        # --- Notified Body Name ---
        nb_name_patterns = [
            r"(bureau\s+veritas[^,\n]*)",
            r"(tuv\s+[a-zçşığüö\s]+(?:gmbh|ag|ltd)?[^,\n]*)",
            r"(sgs\s+[a-zçşığüö\s]+(?:gmbh|ltd|inc)?[^,\n]*)",
            r"(dekra\s+[a-zçşığüö\s]+(?:gmbh|ag)?[^,\n]*)",
            r"(intertek\s+[a-zçşığüö\s]+(?:ltd|gmbh|inc)?[^,\n]*)",
            r"(bsi\s+[a-zçşığüö\s]+(?:ltd|gmbh)?[^,\n]*)",
            r"(lloyd's\s+register[^,\n]*)",
            r"(dnv\s+[a-zçşığüö\s]*(?:gl)?[^,\n]*)",
            r"([A-Za-zÇŞİĞÜÖıçşığüö\s&\.]{5,50})\s*(?:ltd|gmbh|inc|corp|ag|certification|testing|inspection|prüfung|notified\s+body)"
        ]
        for pattern in nb_name_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                values["notified_body_name"] = match.group(1).strip()
                break

        # --- Notified Body ID ---
        nb_id_patterns = [
            r"(?:notified\s+body|nb|onaylanmış\s+kuruluş|kimlik|id)\s*[:\-]?\s*([0-9]{4})",
            r"nb\s*([0-9]{4})",
            r"([0-9]{4})\s*(?:notified|onaylanmış)"
        ]
        for pattern in nb_id_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                values["notified_body_id"] = match.group(1).strip()
                break

        # --- Manufacturer Name ---
        manuf_patterns = [
            r"(?:manufacturer|imalatçı|imalatci|üretici|fabrika|fabricant|hersteller|applicant|başvuru\s+sahibi|müracaatçı|company|şirket|firma)\s*[:\-]?\s*([A-Za-zÇŞİĞÜÖıçşığüö\s\.\-&]{5,100})",
            r"(?:we\s+hereby\s+certify\s+that\s+)([A-Za-zÇŞİĞÜÖıçşığüö\s&\.]+)",
            r"(?:this\s+certificate\s+is\s+issued\s+to\s+)([A-Za-zÇŞİĞÜÖıçşığüö\s&\.]+)"
        ]
        for pattern in manuf_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                values["manufacturer_name"] = match.group(1).strip()
                break

        # --- Machine Type/Model ---
        machine_patterns = [
            r"(?:machine\s+type|makine\s+tipi|makine\s+modeli|tipo\s+de\s+máquina|maschinentyp|type\s+de\s+machine)\s*[:\-]?\s*([A-Za-zÇŞİĞÜÖıçşığüö0-9\s\.\-/]{3,80})",
            r"(?:trade\s+name|ticari\s+ad|ticari\s+isim|commercial\s+name|product\s+name|ürün\s+adı)\s*[:\-]?\s*([A-Za-zÇŞİĞÜÖıçşığüö0-9\s\.\-/]{3,80})",
            r"(?:model|modelo|modèle|modell|tip|çeşit)\s*[:\-]?\s*([A-Za-zÇŞİĞÜÖıçşığüö0-9\s\.\-/]{2,50})"
        ]
        for pattern in machine_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                values["machine_type"] = match.group(1).strip()
                break

        # --- Certificate Number ---
        cert_patterns = [
            r"(?:certificate\s+number|belge\s+numarası|sertifika\s+no|cert\.\s*no\.?|number|nummer|número|ref|reference)\s*[:\-]?\s*([A-Za-zÇŞİĞÜÖıçşığüö0-9\-/\.]{5,30})",
            r"(?:number|nummer|número|ref|reference)\s*[:\-]?\s*([A-Za-zÇŞİĞÜÖıçşığüö0-9\-/\.]{8,30})"
        ]
        for pattern in cert_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                cert_num = match.group(1).strip()
                if len(cert_num) >= 5:
                    values["certificate_number"] = cert_num
                    break

        # --- Issue Date ---
        date_patterns = [
            r"(?:date|tarih|datum|fecha|düzenlenme\s+tarihi|issue\s+date)\s*[:\-]?\s*([0-9]{1,2}[\.\/\-][0-9]{1,2}[\.\/\-][0-9]{2,4})",
            r"([0-9]{1,2}\s+[A-Za-zÇŞİĞÜÖıçşığüö]{3,9}\s+[0-9]{4})",
            r"([0-9]{4}[\.\/\-][0-9]{1,2}[\.\/\-][0-9]{1,2})"
        ]
        for pattern in date_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                values["issue_date"] = match.group(1).strip()
                break

        # --- Serial Number ---
        serial_patterns = [
            r"(?:serial\s+number|seri\s+numarası|seri\s+no|s/n|sn|série|seriennummer|sıra\s+no|üretim\s+no)\s*[:\-]?\s*([A-Za-zÇŞİĞÜÖıçşığüö0-9\-/\.]{2,25})"
        ]
        for pattern in serial_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                values["serial_number"] = match.group(1).strip()
                break

        # --- Applied Standards ---
        standards = re.findall(
            r"(?:EN\s*ISO\s*[0-9]{3,5}[\-:]*[0-9]*[\-:]*[0-9]*|EN\s*[0-9]{3,5}[\-:]*[0-9]*[\-:]*[0-9]*|ISO\s*[0-9]{3,5}[\-:]*[0-9]*[\-:]*[0-9]*|IEC\s*[0-9]{3,5}[\-:]*[0-9]*[\-:]*[0-9]*)",
            text, re.IGNORECASE)
        values["applied_standards"] = list(set(standards))

        # --- Directive Reference ---
        if re.search(r"2006/42/EC|2006\/42\/EC|machinery\s+directive|makine\s+direktifi", text, re.IGNORECASE):
            values["directive_reference"] = "2006/42/EC"

        return values

    # ---------- Öneriler ----------
    def generate_recommendations(self, analysis_results: Dict[str, Dict[str, ATTipIncelemeResult]],
                                 scores: Dict[str, Any]) -> List[str]:
        recommendations = []
        if scores["critical_missing"]:
            recommendations.append("🚨 KRİTİK EKSİKLİKLER - BELGE GEÇERSİZDİR!")
            recommendations.append("⚠️ 2006/42/EC Ek IX'a göre aşağıdaki bilgilerden biri eksikse belge geçersizdir:")
            for missing in scores["critical_missing"]:
                recommendations.append(f"  ❌ {missing}")
            recommendations.append("")
        for category, score_data in scores["category_scores"].items():
            if score_data["percentage"] < 100:
                missing_items = []
                for criterion_name, result in analysis_results[category].items():
                    if result.is_critical and not result.found:
                        missing_items.append(result.details['description'])
                if missing_items:
                    recommendations.append(f"🚨 {category} - Kritik Eksikler:")
                    for item in missing_items:
                        recommendations.append(f"  ❌ {item}")
        total_percentage = scores["percentage"]
        critical_missing_count = len(scores["critical_missing"])
        if critical_missing_count > 0:
            recommendations.append("🔴 SONUÇ: BELGE GEÇERSİZDİR")
            recommendations.append("⚖️ Hukuki Durum: 2006/42/EC Ek IX gereksinimlerini karşılamıyor")
            recommendations.append("🔧 Acil Eylem: Eksik bilgileri tamamlayarak yeni belge düzenlenmeli")
        elif total_percentage >= 90:
            recommendations.append("✅ SONUÇ: BELGE TAM UYGUNLUKTA")
            recommendations.append("⚖️ Hukuki Durum: 2006/42/EC Ek IX gereksinimlerini tam karşılıyor")
            recommendations.append("📋 Durum: AT Tip İncelemesi Belgesi hukuken geçerlidir")
        elif total_percentage >= 80:
            recommendations.append("🟡 SONUÇ: BELGE KABUL EDİLEBİLİR")
            recommendations.append("⚖️ Hukuki Durum: Temel gereksinimleri karşılıyor")
            recommendations.append("💡 Öneri: Teknik detaylar geliştirilebilir")
        else:
            recommendations.append("🟠 SONUÇ: BELGE YETERSİZ")
            recommendations.append("⚖️ Hukuki Durum: Önemli eksiklikler mevcut")
            recommendations.append("🔍 Öneri: Belge gözden geçirilmeli")
        return recommendations

    # ---------- Ana Analiz ----------
    def analyze_type_examination_certificate(self, pdf_path: str) -> Dict[str, Any]:
        logging.info("Type-Examination Certificate analysis starting...")
        try:
            text = self.extract_text_from_pdf(pdf_path)
            if len(text.strip()) < 50:
                return {"error": "PDF'den yeterli metin çıkarılamadı. Dosya bozuk olabilir veya sadece resim içeriyor olabilir.", "text_length": len(text)}
            detected_language = self.detect_language(text)
            logging.info(f"Detected language: {detected_language}")
            extracted_values = self.extract_specific_values(text)
            category_analyses = {}
            for category in self.criteria_weights.keys():
                category_analyses[category] = self.analyze_criteria(text, category)
            scoring = self.calculate_scores(category_analyses)
            recommendations = self.generate_recommendations(category_analyses, scoring)
            percentage = scoring["percentage"]
            has_critical_missing = len(scoring["critical_missing"]) > 0
            if has_critical_missing:
                status = "INVALID"
                status_tr = "GEÇERSİZ"
            elif percentage >= 90:
                status = "FULLY_COMPLIANT"
                status_tr = "TAM UYGUNLUK"
            elif percentage >= 80:
                status = "ACCEPTABLE"
                status_tr = "KABUL EDİLEBİLİR"
            elif percentage >= 70:
                status = "CONDITIONAL"
                status_tr = "KOŞULLU"
            else:
                status = "INSUFFICIENT"
                status_tr = "YETERSİZ"
            return {
                "analysis_date": datetime.now().isoformat(),
                "file_info": {
                    "filename": os.path.basename(pdf_path),
                    "text_length": len(text),
                    "detected_language": detected_language
                },
                "extracted_values": extracted_values,
                "category_analyses": category_analyses,
                "scoring": scoring,
                "recommendations": recommendations,
                "summary": {
                    "total_score": scoring["total_score"],
                    "percentage": percentage,
                    "status": status,
                    "status_tr": status_tr,
                    "critical_missing_count": len(scoring["critical_missing"]),
                    "report_type": "AT Tip İncelemesi Belgesi (EC Type-Examination Certificate)"
                }
            }
        except Exception as e:
            logging.error(f"Analysis error: {e}")
            return {"error": f"Analiz sırasında hata oluştu: {str(e)}", "analysis_date": datetime.now().isoformat()}

# ---------- Rapor Yazdırma ----------
def print_type_examination_report(report: Dict[str, Any]):
    if "error" in report:
        print(f"❌ Hata: {report['error']}")
        return
    print("\n📊 AT TİP İNCELEMESİ BELGESİ ANALİZİ")
    print("=" * 65)
    print(f"📅 Analiz Tarihi: {report['analysis_date']}")
    print(f"🔍 Tespit Edilen Dil: {report['file_info']['detected_language'].upper()}")
    print(f"📋 Toplam Puan: {report['summary']['total_score']}/100")
    print(f"📈 Yüzde: %{report['summary']['percentage']:.0f}")
    print(f"🎯 Durum: {report['summary']['status_tr']}")
    print(f"⚠️ Kritik Eksik Sayısı: {report['summary']['critical_missing_count']}")
    print(f"📄 Rapor Türü: {report['summary']['report_type']}")

    print("\n📋 ÇIKARILAN TEMEL BİLGİLER")
    print("-" * 45)
    extracted_values = report['extracted_values']
    display_names = {
        "notified_body_name": "Onaylanmış Kuruluş Adı",
        "notified_body_address": "Onaylanmış Kuruluş Adresi",
        "notified_body_id": "Kuruluş Kimlik No",
        "manufacturer_name": "İmalatçı Adı",
        "manufacturer_address": "İmalatçı Adresi",
        "machine_trade_name": "Makinenin Ticari Adı",
        "machine_type": "Makine Tipi",
        "machine_model": "Model",
        "serial_number": "Seri No",
        "certificate_number": "Belge Numarası",
        "issue_date": "Düzenlenme Tarihi",
        "validity_date": "Geçerlilik Süresi",
        "directive_reference": "Direktif Atfı",
        "applied_standards": "Uygulanan Standartlar",
        "authorized_person": "Yetkili Kişi"
    }
    for key, value in extracted_values.items():
        if key in display_names:
            if key == "applied_standards":
                standards_str = ", ".join(value) if value else "Bulunamadı"
                print(f"{display_names[key]}: {standards_str}")
            else:
                print(f"{display_names[key]}: {value}")

    print("\n📊 KATEGORİ PUANLARI")
    print("-" * 45)
    for category, score_data in report['scoring']['category_scores'].items():
        status_icon = "🟢" if score_data['percentage'] == 100 else "🟡" if score_data['percentage'] >= 80 else "🔴"
        print(f"{status_icon} {category}")
        print(f"   Puan: {score_data['normalized']}/{score_data['max_weight']} (%{score_data['percentage']:.0f})")

    print("\n🚨 KRİTİK EKSİKLİKLER (GEÇERSİZLİK SEBEPLERİ)")
    print("-" * 50)
    if report['scoring']['critical_missing']:
        print("⚠️ Aşağıdaki bilgiler eksik olduğu için belge GEÇERSİZDİR:")
        for missing in report['scoring']['critical_missing']:
            print(f"❌ {missing}")
    else:
        print("✅ Kritik eksiklik bulunamadı - Belge temel gereksinimleri karşılıyor")

    print("\n💡 DEĞERLENDİRME VE ÖNERİLER")
    print("-" * 45)
    for recommendation in report['recommendations']:
        print(recommendation)

    print("\n📋 2006/42/EC EK IX UYGUNLUK DEĞERLENDİRMESİ")
    print("=" * 65)
    if report['summary']['status'] == "INVALID":
        print("🚨 SONUÇ: BELGE GEÇERSİZDİR")
        print(f"❌ Kritik eksiklikler: {report['summary']['critical_missing_count']} adet")
        print("⚖️ Hukuki Durum: 2006/42/EC Ek IX gereksinimlerini karşılamıyor")
        print("🔧 Eylem: Belge yeniden düzenlenmeli veya eksiklikler giderilmeli")
    elif report['summary']['status'] == "FULLY_COMPLIANT":
        print("✅ SONUÇ: BELGE TAM UYGUNLUKTA")
        print(f"🌟 Toplam Başarı: %{report['summary']['percentage']:.0f}")
        print("⚖️ Hukuki Durum: 2006/42/EC Ek IX gereksinimlerini tam karşılıyor")
        print("📋 Durum: AT Tip İncelemesi Belgesi hukuken geçerlidir")
    elif report['summary']['status'] == "ACCEPTABLE":
        print("🟡 SONUÇ: BELGE KABUL EDİLEBİLİR")
        print(f"📈 Toplam Başarı: %{report['summary']['percentage']:.0f}")
        print("⚖️ Hukuki Durum: Temel gereksinimleri karşılıyor")
        print("💡 Öneri: Teknik detaylar geliştirilebilir")
    else:
        print("❌ SONUÇ: BELGE YETERSİZ")
        print(f"⚠️ Toplam Başarı: %{report['summary']['percentage']:.0f}")
        print("⚖️ Hukuki Durum: Direktif gereksinimlerini karşılamıyor")
        print("🚨 Öneri: Kapsamlı gözden geçirme gerekli")

    print("\n📚 HUKUKİ DAYANAK")
    print("-" * 20)
    print("• 2006/42/EC Makine Direktifi")
    print("• Ek IX - AT Tip İncelemesi Prosedürü")
    print("• Onaylanmış Kuruluş Yükümlülükleri")

# ---------- Main ----------
def main():
    import sys
    test_files = [
        r"C:\Users\nuvo_teknik_2\Desktop\PILZ DOCUMENTS\4.2 AT Tip İnceleme Sertifikası\dirinler-makina-cdsh-11.pdf"
    ]
    import glob
    cert_files = (glob.glob("*Type*Examination*.pdf") +
                  glob.glob("*EC*Type*.pdf") +
                  glob.glob("*Certificate*.pdf") +
                  glob.glob("*TIP*INCELEME*.pdf") +
                  glob.glob("*TYPE*EXAM*.pdf"))
    test_files.extend(cert_files)
    selected_file = None
    for file in test_files:
        if '*' in file:
            matches = glob.glob(file)
            if matches:
                selected_file = matches[0]
                break
        elif os.path.exists(file):
            selected_file = file
            break
    if selected_file:
        print(f"🔍 Analiz edilen dosya: {selected_file}")
    else:
        print("❌ Hiçbir AT Tip İncelemesi Belgesi bulunamadı")
        print("📁 Lütfen EC Type-Examination Certificate dosyasının proje klasöründe olduğundan emin olun.")
        print("🔍 Desteklenen dosya formatları:")
        print("   • *Type*Examination*.pdf")
        print("   • *EC*Type*.pdf")
        print("   • *Certificate*.pdf")
        print("   • *TIP*INCELEME*.pdf")
        print("   • *TYPE*EXAM*.pdf")
        sys.exit(1)
    analyzer = ATTipIncelemeAnalyzer()
    report = analyzer.analyze_type_examination_certificate(selected_file)
    print_type_examination_report(report)

if __name__ == "__main__":
    main()