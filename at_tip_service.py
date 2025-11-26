"""
AT Tip İnceleme Sertifikası Analiz Servisi
==========================================
Endpoint: POST /api/at-type-cert-report
Health: GET /api/health
"""

import PyPDF2
import cv2
import numpy as np
from PIL import Image
import pdf2image
import pytesseract
from flask import Flask, request, jsonify
import os
import re
from datetime import datetime
from werkzeug.utils import secure_filename
import logging
from dataclasses import dataclass
from typing import Dict, Any, List
from docx import Document

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from langdetect import detect
    LANG_DETECT = True
except:
    LANG_DETECT = False

@dataclass
class ATTipIncelemeResult:
    criteria_name: str
    found: bool
    content: str
    score: int
    max_score: int
    is_critical: bool
    details: Dict[str, Any]

class ATTipIncelemeAnalyzer:
    def __init__(self):
        logger.info("AT Type-Examination Certificate analysis system starting...")
        
        # ESKİ HALDEKİ DOĞRU WEIGHTS (2. dokümandan)
        self.criteria_weights = {
            "Onaylanmış Kuruluş Bilgileri": 20,
            "Başvuru Sahibi/İmalatçı Bilgileri": 20,
            "Makine Tanımı": 15,
            "İncelenen Tip Tanımı": 10,
            "Uygulanan Hükümler": 15,
            "Değerlendirme Sonucu": 10,
            "Belge Geçerlilik Bilgileri": 10
        }

        # ESKİ HALDEKİ DETAYLI TÜRKÇE & İNGİLİZCE KARMA DESENLER
        self.criteria_details = {
            "Onaylanmış Kuruluş Bilgileri": {
                "kurulusun_adi": {
                    "pattern": r"(?:notified\s+body|onaylanmış\s+kuruluş|onaylı\s+kuruluş|nb|bureau\s+veritas|tuv|sgs|dekra|intertek|bsi|lloyd's\s+register|dnv|kiwa|icim|csi|mts)[\s\w]*([A-Za-zÇŞİĞÜÖıçşığüö\s\.\-&]{5,80})|([A-Za-zÇŞİĞÜÖıçşığüö\s&\.]{5,50})\s*(?:ltd|gmbh|inc|corp|ag|certification|testing|inspection|prüfung|notified\s+body)",
                    "weight": 7,
                    "critical": True,
                    "description": "Onaylanmış kuruluşun adı"
                },
                "kurulusun_adresi": {
                    "pattern": r"(?:address|adres|adresi|konumu|yeri|sede|adresse)\s*[:\-]?\s*([A-Za-zÇŞİĞÜÖıçşığüö0-9\s\.\-/,&]{20,150})|(?:street|road|avenue|str\.|strasse|calle|via|cadde|sokak)[\s\w]*([A-Za-zÇŞİĞÜÖıçşığüö0-9\s\.\-/,&]{15,100})|(?:[0-9]{1,5}\s+[A-Za-zÇŞİĞÜÖıçşığüö\s]{5,50}(?:street|road|avenue|str\.|strasse|cadde|sokak))|(?:D-[0-9]{5}\s+[A-Za-zÇŞİĞÜÖıçşığüö]+)",
                    "weight": 6,
                    "critical": True,
                    "description": "Onaylanmış kuruluşun adresi"
                },
                "kimlik_numarasi": {
                    "pattern": r"(?:notified\s+body|nb|identification|kimlik|id|number|numarası|no|nummer|número)\s*[:\-]?\s*([0-9]{4})|(?:nb\s*[0-9]{4})|([0-9]{4})(?:\s*(?:notified|onaylanmış))",
                    "weight": 7,
                    "critical": True,
                    "description": "Onaylanmış kuruluş kimlik numarası (4 haneli)"
                }
            },
            "Başvuru Sahibi/İmalatçı Bilgileri": {
                "imalatci_adi": {
                    "pattern": r"(?:manufacturer|imalatçı|imalatci|fabrika|üretici|fabricant|hersteller|applicant|başvuru\s+sahibi|müracaatçı|company|şirket|firma|üretim\s+yeri)[\s:]*([A-Za-zÇŞİĞÜÖıçşığüö\s\.\-&]{5,100})|(?:we\s+hereby\s+certify\s+that\s+)([A-Za-zÇŞİĞÜÖıçşığüö\s&\.]+)|(?:this\s+certificate\s+is\s+issued\s+to\s+)([A-Za-zÇŞİĞÜÖıçşığüö\s&\.]+)",
                    "weight": 10,
                    "critical": True,
                    "description": "İmalatçı veya yetkili temsilcinin adı"
                },
                "imalatci_adres": {
                    "pattern": r"(?:manufacturer\s+address|imalatçı\s+adres|imalatci\s+adres|adresse\s+du\s+fabricant|herstelleradresse|üretici\s+adresi)\s*[:\-]?\s*([A-Za-zÇŞİĞÜÖıçşığüö0-9\s\.\-/,&]{20,150})|(?:located\s+at|registered\s+at|address|adres|konum|yer)\s*[:\-]?\s*([A-Za-zÇŞİĞÜÖıçşığüö0-9\s\.\-/,&]{15,120})",
                    "weight": 10,
                    "critical": True,
                    "description": "İmalatçı veya yetkili temsilcinin tam adresi"
                }
            },
            "Makine Tanımı": {
                "ticari_ad_tip": {
                    "pattern": r"(?:trade\s+name|ticari\s+ad|ticari\s+isim|commercial\s+name|product\s+name|denomination|type|tip|model|bezeichnung|ürün\s+adı|makine\s+adı)\s*[:\-]?\s*([A-Za-zÇŞİĞÜÖıçşığüö0-9\s\.\-/]{3,80})|(?:machine\s+type|makine\s+tipi|makine\s+modeli)\s*[:\-]?\s*([A-Za-zÇŞİĞÜÖıçşığüö0-9\s\.\-/]{3,80})",
                    "weight": 8,
                    "critical": True,
                    "description": "Makinenin ticari adı, tipi, modeli"
                },
                "seri_numarasi": {
                    "pattern": r"(?:serial\s+number|seri\s+numarası|seri\s+no|s/n|sn|série|seriennummer|sıra\s+no|üretim\s+no)\s*[:\-]?\s*([A-Za-zÇŞİĞÜÖıçşığüö0-9\-/\.]{2,25})|(?:serial)\s*[:\-]?\s*([A-Za-zÇŞİĞÜÖıçşığüö0-9\-/\.]{2,25})",
                    "weight": 4,
                    "critical": True,
                    "description": "Seri numarası veya tanımlamayı sağlayan bilgiler"
                },
                "varyantlar": {
                    "pattern": r"(?:variant|varyant|version|versions|sürüm|model\s+variants|configuration|konfigürasyon|seçenekler)\s*[:\-]?\s*([A-Za-zÇŞİĞÜÖıçşığüö0-9\s\.\-/,]{5,100})",
                    "weight": 3,
                    "critical": False,
                    "description": "Varyantlar veya versiyonlar (varsa)"
                }
            },
            "İncelenen Tip Tanımı": {
                "detayli_tanim": {
                    "pattern": r"(?:detailed\s+description|ayrıntılı\s+tanım|detaylı\s+açıklama|description\s+of\s+the\s+machine|machine\s+description|technical\s+description|makine\s+açıklaması|teknik\s+tanım)\s*[:\-]?\s*([A-Za-zÇŞİĞÜÖıçşığüö0-9\s\.\-/,&]{10,200})",
                    "weight": 4,
                    "critical": True,
                    "description": "İncelenen tipin ayrıntılı tanımı"
                },
                "teknik_dosya_atif": {
                    "pattern": r"(?:technical\s+file|teknik\s+dosya|teknik\s+evrak|technical\s+documentation|dossier\s+technique|technische\s+unterlage|documentation|reference|belge|dokümantasyon)\s*[:\-]?\s*([A-Za-zÇŞİĞÜÖıçşığüö0-9\s\.\-/]{5,50})|(?:according\s+to|göre|in\s+accordance\s+with)\s+(?:technical\s+file|documentation|teknik\s+dosya)",
                    "weight": 3,
                    "critical": True,
                    "description": "İlgili teknik dosyaya atıf"
                },
                "resim_plan_sema": {
                    "pattern": r"(?:drawing|plan|schema|şema|resim|picture|figure|şekil|diagram|blueprint|çizim|photos|fotoğraf|plan|poz|vaziyet\s+planı|montaj\s+resmi)\s*[:\-]?\s*([A-Za-zÇŞİĞÜÖıçşığüö0-9\s\.\-/]{3,50})|(?:as\s+shown\s+in|gösterildiği\s+gibi|according\s+to\s+drawing|çizime\s+göre)",
                    "weight": 3,
                    "critical": False,
                    "description": "Resim, plan, şema, parça listeleri"
                }
            },
            "Uygulanan Hükümler": {
                "direktif_atif": {
                    "pattern": r"(?:2006/42/EC|2006\/42\/EC|machinery\s+directive|makine\s+direktifi|directive\s+2006/42|machine\s+safety\s+directive|makine\s+emniyet\s+direktifi)",
                    "weight": 8,
                    "critical": True,
                    "description": "2006/42/EC direktif maddelerine atıf"
                },
                "uyumlastirilmis_standartlar": {
                    "pattern": r"(?:EN\s*ISO\s*[0-9]{3,5}[\-:]*[0-9]*[\-:]*[0-9]*|EN\s*[0-9]{3,5}[\-:]*[0-9]*[\-:]*[0-9]*|ISO\s*[0-9]{3,5}[\-:]*[0-9]*[\-:]*[0-9]*|IEC\s*[0-9]{3,5}[\-:]*[0-9]*[\-:]*[0-9]*)",
                    "weight": 5,
                    "critical": True,
                    "description": "Uyumlaştırılmış standartlar (EN ISO, EN IEC vb.)"
                },
                "esdeger_cozumler": {
                    "pattern": r"(?:equivalent\s+solution|eşdeğer\s+çözüm|alternative\s+solution|other\s+technical\s+solution|diğer\s+teknik\s+çözüm|non-harmonised|harmonize\s+olmayan|alternatif\s+çözüm)",
                    "weight": 2,
                    "critical": False,
                    "description": "Eşdeğer çözümlerin açıklaması (varsa)"
                }
            },
            "Değerlendirme Sonucu": {
                "uygunluk_ifadesi": {
                    "pattern": r"(?:complies\s+with|uygun|conform|conforms\s+to|in\s+compliance|meets\s+the\s+requirements|requirements\s+of|satisfies|karşılar|uygunluğu|conformity|compliance|uygundur|uygun\s+olduğu|uygunluk\s+ifadesi)",
                    "weight": 6,
                    "critical": True,
                    "description": "Direktif hükümlerine uygunluk ifadesi"
                },
                "test_muayene_ozet": {
                    "pattern": r"(?:test|muayene|examination\s+carried\s+out|inspection|assessment|değerlendirme|inceleme|kontrolü|yapılan\s+testler|performed\s+tests|evaluated|examined|test\s+sonuçları|muayene\s+raporu)",
                    "weight": 4,
                    "critical": True,
                    "description": "Yapılan testler/muayeneler/hesaplamaların özeti"
                }
            },
            "Belge Geçerlilik Bilgileri": {
                "duzenleme_tarihi": {
                    "pattern": r"(?:date|tarih|datum|fecha|düzenlenme\s+tarihi|issue\s+date)\s*[:\-]?\s*([0-9]{1,2}[\.\/\-][0-9]{1,2}[\.\/\-][0-9]{2,4})|([0-9]{1,2}\s+[A-Za-zÇŞİĞÜÖıçşığüö]{3,9}\s+[0-9]{4})|([0-9]{4}[\.\/\-][0-9]{1,2}[\.\/\-][0-9]{1,2})",
                    "weight": 3,
                    "critical": True,
                    "description": "Belgenin düzenlenme tarihi"
                },
                "belge_numarasi": {
                    "pattern": r"(?:certificate\s+number|belge\s+numarası|sertifika\s+no|cert\.\s*no\.?|number|nummer|número|ref|reference)\s*[:\-]?\s*([A-Za-zÇŞİĞÜÖıçşığüö0-9\-/\.]{5,30})|(?:cert\.\s*no\.?)\s*[:\-]?\s*([A-Za-zÇŞİĞÜÖıçşığüö0-9\-/\.]{5,30})",
                    "weight": 4,
                    "critical": True,
                    "description": "Belge numarası"
                },
                "gecerlilik_suresi": {
                    "pattern": r"(?:valid\s+until|geçerli|validity|expires|expiry\s+date|son\s+geçerlilik|until|bis|geçerlilik\s+süresi)\s*[:\-]?\s*([0-9]{1,2}[\.\/\-][0-9]{1,2}[\.\/\-][0-9]{2,4})|(?:this\s+certificate\s+is\s+valid)|(?:remains\s+valid)",
                    "weight": 2,
                    "critical": False,
                    "description": "Geçerlilik süresi (varsa)"
                },
                "yetkili_imza": {
                    "pattern": r"(?:signed\s+by|imzalayan|signature|imza|authorized\s+by|yetkili\s+temsilci|responsible\s+person|sorumlu\s+kişi|signatory|signed\s+for|on\s+behalf|imza\s+yetkilisi)",
                    "weight": 1,
                    "critical": True,
                    "description": "Yetkili temsilcinin imzası"
                }
            }
        }

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """PDF'den metin çıkarımı - PyPDF2 + OCR fallback"""
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    text += page.extract_text() or ""
            
            logger.info(f"PyPDF2 extracted {len(text)} characters")
            
            if len(text.strip()) < 50:
                logger.info("Insufficient text with PyPDF2, trying OCR...")
                pages = pdf2image.convert_from_path(pdf_path, dpi=200)
                ocr_text = ""
                
                for i, page in enumerate(pages, 1):
                    try:
                        page_text = pytesseract.image_to_string(page, lang='tur+eng+deu+fra+spa')
                        ocr_text += page_text + "\n"
                        logger.info(f"OCR extracted {len(page_text)} characters from page {i}")
                    except Exception as e:
                        logger.warning(f"OCR failed for page {i}: {e}")
                        continue
                
                if len(ocr_text.strip()) > len(text.strip()):
                    text = ocr_text
                    logger.info(f"OCR total text length: {len(text)}")
        
        except Exception as e:
            logger.error(f"Error extracting text: {e}")
            return ""
        
        return text

    def extract_text_from_docx(self, docx_path: str) -> str:
        """DOCX'den metin çıkarımı"""
        try:
            doc = Document(docx_path)
            return "\n".join([p.text for p in doc.paragraphs])
        except Exception as e:
            logger.error(f"DOCX hatası: {e}")
            return ""

    def extract_text_from_txt(self, txt_path: str) -> str:
        """TXT'den metin çıkarımı"""
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception:
            return ""

    def detect_language(self, text: str) -> str:
        if not LANG_DETECT:
            return 'en'
        try:
            return detect(text[:500].strip()) if len(text.strip()) >= 50 else "en"
        except:
            return 'en'
        
    def analyze_criteria(self, text: str, category: str) -> Dict[str, ATTipIncelemeResult]:
        """Kriterleri analiz et"""
        results = {}
        criteria = self.criteria_details.get(category, {})
        
        for criterion_name, criterion_data in criteria.items():
            pattern = criterion_data["pattern"]
            weight = criterion_data["weight"]
            is_critical = criterion_data["critical"]
            description = criterion_data["description"]
            
            matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
            
            if matches:
                # Clean up matches and get the best one
                clean_matches = []
                for match in matches:
                    if isinstance(match, tuple):
                        # For groups in regex, take the first non-empty group
                        clean_match = next((m for m in match if m.strip()), "")
                    else:
                        clean_match = str(match)
                    
                    if clean_match.strip():
                        clean_matches.append(clean_match.strip())
                
                if clean_matches:
                    content = f"Bulundu: {clean_matches[0][:60]}..."
                    found = True
                    score = weight  # Full points
                else:
                    content = "Eşleşme bulundu ama değer çıkarılamadı"
                    found = True
                    score = int(weight * 0.5)  # Partial points
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
                details={
                    "description": description,
                    "pattern_used": pattern,
                    "matches_count": len(matches) if matches else 0,
                    "raw_matches": matches[:3] if matches else []
                }
            )
        
        return results

    def calculate_scores(self, analysis_results: Dict[str, Dict[str, ATTipIncelemeResult]]) -> Dict[str, Any]:
        """Puanlama hesaplama"""
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
                percentage = 0
                normalized_score = 0
            
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

    def extract_specific_values(self, text: str) -> Dict[str, Any]:
        """Spesifik değerleri çıkar - ESKİ HALDEKİ DETAYLI VERSİYON"""
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

        # Notified Body Name - Çoklu pattern
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

        # Notified Body ID
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

        # Manufacturer Name
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

        # Machine Type/Model
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

        # Certificate Number
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

        # Issue Date
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

        # Serial Number
        serial_patterns = [
            r"(?:serial\s+number|seri\s+numarası|seri\s+no|s/n|sn|série|seriennummer|sıra\s+no|üretim\s+no)\s*[:\-]?\s*([A-Za-zÇŞİĞÜÖıçşığüö0-9\-/\.]{2,25})"
        ]
        for pattern in serial_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                values["serial_number"] = match.group(1).strip()
                break

        # Applied Standards
        standards = re.findall(
            r"(?:EN\s*ISO\s*[0-9]{3,5}[\-:]*[0-9]*[\-:]*[0-9]*|EN\s*[0-9]{3,5}[\-:]*[0-9]*[\-:]*[0-9]*|ISO\s*[0-9]{3,5}[\-:]*[0-9]*[\-:]*[0-9]*|IEC\s*[0-9]{3,5}[\-:]*[0-9]*[\-:]*[0-9]*)",
            text, re.IGNORECASE)
        values["applied_standards"] = list(set(standards))

        # Directive Reference
        if re.search(r"2006/42/EC|2006\/42\/EC|machinery\s+directive|makine\s+direktifi", text, re.IGNORECASE):
            values["directive_reference"] = "2006/42/EC"

        return values

    def generate_recommendations(self, analysis_results: Dict[str, Dict[str, ATTipIncelemeResult]], 
                                scores: Dict[str, Any]) -> List[str]:
        """Öneriler oluştur - ESKİ HALDEKİ DETAYLI VERSİYON"""
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

    def analyze_type_examination_certificate(self, file_path: str) -> Dict[str, Any]:
        """Ana analiz fonksiyonu"""
        logger.info("Type-Examination Certificate analysis starting...")
        
        try:
            if not os.path.exists(file_path):
                return {"error": f"Dosya bulunamadı: {file_path}"}
            
            file_ext = os.path.splitext(file_path)[1].lower()
            
            # Metin çıkarımı
            if file_ext == '.pdf':
                text = self.extract_text_from_pdf(file_path)
            elif file_ext in ['.docx', '.doc']:
                text = self.extract_text_from_docx(file_path)
            elif file_ext == '.txt':
                text = self.extract_text_from_txt(file_path)
            else:
                return {"error": f"Desteklenmeyen dosya formatı: {file_ext}"}
            
            if len(text.strip()) < 50:
                return {
                    "error": "Dosyadan yeterli metin çıkarılamadı",
                    "text_length": len(text)
                }
            
            detected_language = self.detect_language(text)
            logger.info(f"Detected language: {detected_language}")
            
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
                    "filename": os.path.basename(file_path),
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
                    "report_type": "AT_TIP_INCELEME_SERTIFIKASI"
                }
            }
        
        except Exception as e:
            logger.error(f"Analysis error: {e}")
            return {
                "error": f"Analiz sırasında hata oluştu: {str(e)}",
                "analysis_date": datetime.now().isoformat()
            }
def validate_document_server(text):
    """Server kodunda doküman validasyonu - AT Type Certificate için"""
    critical_terms = [
        # AT Tip temel terimleri (en az 1 tane olmalı)
        ["inceleme", "examination", "sertifika", "certificate", "belge", "document", "at tip", "ec type"],
        
        # Makine direktifi terimleri (en az 1 tane olmalı)  
        ["direktif", "directive", "makine", "machinery", "2006/42/ec", "42/ec", "ek ix", "annex ix"],
        
        # Onaylanmış kuruluş terimleri (mutlaka olmalı)
        ["onaylanmış", "notified", "kuruluş", "body", "notified body", "onaylanmış kuruluş"],
        
        # Belge geçerlilik terimleri (en az 1 tane olmalı)
        ["geçerli", "valid", "yetki", "authority", "onay", "approval", "tarih", "date"]
    ]
    
    category_found = []
    for i, category in enumerate(critical_terms):
        found_in_category = False
        for term in category:
            if re.search(rf"\b{term}\b", text, re.IGNORECASE):
                found_in_category = True
                logger.info(f"AT Type Cert Kategori {i+1} bulundu: '{term}'")
                break
        category_found.append(found_in_category)
    
    valid_categories = sum(category_found)
    logger.info(f"Doküman validasyonu: {valid_categories}/4 kritik kategori bulundu")
    return valid_categories >= 3

def check_strong_keywords_first_pages(filepath):
    """İlk 1-2 sayfada özgü kelimeleri OCR ile ara - AT Type Certificate için"""
    strong_keywords = [
        "AT TİP",
        "at tip",
        "ec type",
        "SERTİFİKA",
        "sertifika",
        "certificate",
        "examination",
        "inceleme"
    ]
    
    try:
        pages = pdf2image.convert_from_path(filepath, dpi=300, first_page=1, last_page=1)
        
        all_text = ""
        for page in pages:
            opencv_image = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
            processed = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            text = pytesseract.image_to_string(processed, config='--oem 3 --psm 6 -l eng+tur')
            all_text += text.lower() + " "
        
        found_keywords = []
        for keyword in strong_keywords:
            if re.search(rf"\b{keyword.lower()}\b", all_text):
                found_keywords.append(keyword)
        
        logger.info(f"İlk sayfa kontrol: {len(found_keywords)} özgü kelime: {found_keywords}")
        return len(found_keywords) >= 1
        
    except Exception as e:
        logger.warning(f"İlk sayfa kontrol hatası: {e}")
        return False

def check_excluded_keywords_first_pages(filepath):
    """İlk 1-2 sayfada istenmeyen rapor türlerinin kelimelerini ara"""
    excluded_keywords = [
        # HRC raporu
        "hrc", "cobot", "robot", "çarpışma", "collaborative", "kolaboratif", "sd conta",
        
        # Elektrik devre şeması
        "elektrik", "devre", "şema", "circuit", "electrical", "voltage", "amper", "ohm", "enclosure", "wrp-", "light curtain", "contactors", "controller",
        
        # Espe raporu  
        "espe",
        
        # Hidrolik devre şeması
        "hidrolik", "HİDROLİK", "hydraulic", "hidrolik yağ", "hydraulic oil", "iso 1219", "1219", "teknik resim", "tasarım",
        
        # Gürültü ölçüm raporu
        "gürültü", "noise", "ses", "sound", "decibel", "db", "akustik", "acoustic",
        
        # Manuel/kullanma kılavuzu
        "kullanma", "kılavuz", "manual", "instruction", "talimat", "guide", "kılavuzu",
        
        # LOTO raporu
        "loto",
        
        # LVD raporu
        "lvd", "TOPRAKLAMA SÜREKLİLİK", "topraklama süreklilik", "TOPRAKLAMA İLETKENLERİ", "topraklama iletkenleri",
        
        # AT uygunluk beyanı (declaration, NOT certificate)
        "uygunluk", "beyan", "conformity", "declaration", "declare",
        
        # İSG periyodik kontrol
        "isg", "periyodik", "kontrol", "periodic", "inspection", "denetim",
        
        # Pnömatik devre şeması
        "pnömatik", "pneumatic", "lubricator", "inflate", "psi", "bar", "oil", "regis", "r102", "regulator", "dump valve",
        
        # Montaj talimatları
        "montaj", "assembly",
        
        # EN 60204-1 topraklama raporu
        "topraklama direnci", "grounding", "earthing", "60204", "topraklama", "TOPRAKLAMA DİRENCİ",
        
        # Bakım talimatları
        "bakım", "maintenance", "servis", "service", "bakim", "MAINTENCE",
        
        # Mekanik titreşim raporu
        "titreşim", "vibration", "mekanik",
        
        # Aydınlatma
        "aydınlatma", "lighting", "illumination", "lux", "lümen", "lumen", "ts en 12464", "en 12464", "ışık", "ışık şiddeti"
    ]
    
    try:
        pages = pdf2image.convert_from_path(filepath, dpi=200, first_page=1, last_page=2)
        
        all_text = ""
        for page in pages:
            opencv_image = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
            processed = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            text = pytesseract.image_to_string(processed, config='--oem 3 --psm 6 -l eng+tur')
            all_text += text.lower() + " "
        
        found_excluded = []
        for keyword in excluded_keywords:
            if re.search(rf"\b{keyword.lower()}\b", all_text):
                found_excluded.append(keyword)
        
        logger.info(f"İlk sayfa excluded kontrol: {len(found_excluded)} istenmeyen kelime: {found_excluded}")
        return len(found_excluded) >= 1
        
    except Exception as e:
        logger.warning(f"İlk sayfa excluded kontrol hatası: {e}")
        return False

def get_conclusion_message(status, percentage):
    if status == "PASS":
        return f"AT Tip İnceleme Sertifikası 2006/42/EC Ek IX'a uygun (%{percentage:.0f})"
    return f"AT Tip İnceleme Sertifikası direktife uygun değil (%{percentage:.0f})"

def get_main_issues(report):
    issues = []
    if report['scoring']['critical_missing']:
        for item in report['scoring']['critical_missing']:
            issues.append(f"Kritik eksik: {item}")
    return issues[:4]

app = Flask(__name__)

UPLOAD_FOLDER = 'temp_uploads_at_type_cert'
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'doc', 'txt'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/at-type-cert-report', methods=['POST'])
def analyze_at_type_cert_report():
    """AT Type Certificate analiz API endpoint'i - 3 Aşamalı Validasyon"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '' or not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file'}), 400
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        analyzer = ATTipIncelemeAnalyzer()
        file_ext = os.path.splitext(filepath)[1].lower()
        
        # ÜÇ AŞAMALI AT TYPE CERTIFICATE KONTROLÜ
        logger.info(f"Üç aşamalı AT Type Certificate kontrolü başlatılıyor: {filename}")
        
        if file_ext == '.pdf':
            logger.info("Aşama 1: İlk sayfa AT Type Certificate özgü kelime kontrolü...")
            if check_strong_keywords_first_pages(filepath):
                logger.info("✅ Aşama 1 geçti - AT Type Certificate özgü kelimeler bulundu")
            else:
                logger.info("Aşama 2: İlk sayfa excluded kelime kontrolü...")
                if check_excluded_keywords_first_pages(filepath):
                    logger.info("❌ Aşama 2'de excluded kelimeler bulundu - AT Type Certificate değil")
                    try:
                        os.remove(filepath)
                    except:
                        pass
                    return jsonify({
                        'error': 'Invalid document type',
                        'message': 'Bu dosya AT Type Certificate değil (farklı rapor türü tespit edildi). Lütfen AT Type Certificate yükleyiniz.',
                        'details': {
                            'filename': filename,
                            'document_type': 'OTHER_REPORT_TYPE',
                            'required_type': 'AT_TYPE_CERTIFICATE'
                        }
                    }), 400
                else:
                    # AŞAMA 3: PyPDF2 ile tam doküman kontrolü
                    logger.info("Aşama 3: Tam doküman critical terms kontrolü...")
                    try:
                        with open(filepath, 'rb') as pdf_file:
                            pdf_reader = PyPDF2.PdfReader(pdf_file)
                            text = ""
                            for page in pdf_reader.pages:
                                text += page.extract_text() + "\n"
                        
                        if not text or len(text.strip()) < 50:
                            try:
                                os.remove(filepath)
                            except:
                                pass
                            return jsonify({
                                'error': 'Text extraction failed',
                                'message': 'Dosyadan yeterli metin çıkarılamadı'
                            }), 400
                        
                        if not validate_document_server(text):
                            try:
                                os.remove(filepath)
                            except:
                                pass
                            return jsonify({
                                'error': 'Invalid document type',
                                'message': 'Yüklediğiniz dosya AT Type Certificate değil! Lütfen geçerli bir AT Type Certificate yükleyiniz.',
                                'details': {
                                    'filename': filename,
                                    'document_type': 'NOT_AT_TYPE_CERT',
                                    'required_type': 'AT_TYPE_CERTIFICATE'
                                }
                            }), 400
                            
                    except Exception as e:
                        logger.error(f"Aşama 3 hatası: {e}")
                        try:
                            os.remove(filepath)
                        except:
                            pass
                        return jsonify({
                            'error': 'Analysis failed',
                            'message': 'Dosya analizi sırasında hata oluştu'
                        }), 500
        
        elif file_ext in ['.docx', '.doc', '.txt']:
            logger.info(f"DOCX/TXT dosyası için tam doküman kontrolü: {file_ext}")
            text = ""
            if file_ext in ['.docx', '.doc']:
                text = analyzer.extract_text_from_docx(filepath)
            elif file_ext == '.txt':
                text = analyzer.extract_text_from_txt(filepath)
            
            if not text or len(text.strip()) == 0:
                try:
                    os.remove(filepath)
                except:
                    pass
                return jsonify({
                    'error': 'Text extraction failed',
                    'message': 'Dosyadan yeterli metin çıkarılamadı'
                }), 400
            
            if not validate_document_server(text):
                try:
                    os.remove(filepath)
                except:
                    pass
                return jsonify({
                    'error': 'Invalid document type',
                    'message': 'Yüklediğiniz dosya AT Type Certificate değil! Lütfen geçerli bir AT Type Certificate yükleyiniz.',
                    'details': {
                        'filename': filename,
                        'document_type': 'NOT_AT_TYPE_CERT',
                        'required_type': 'AT_TYPE_CERTIFICATE'
                    }
                }), 400
        
        logger.info(f"AT Type Certificate doğrulandı, analiz başlatılıyor: {filename}")
        report = analyzer.analyze_type_examination_certificate(filepath)
        
        try:
            os.remove(filepath)
        except:
            pass
        
        if 'error' in report:
            return jsonify({'error': 'Analysis failed', 'message': report['error']}), 400
        
        overall_percentage = report['summary']['percentage']
        status = "PASS" if overall_percentage >= 70 else "FAIL"
        
        # Extracted values'ı Türkçe key'lerle dönüştür
        extracted_values_tr = {}
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
        
        for eng_key, tr_name in display_names.items():
            if eng_key in report['extracted_values']:
                value = report['extracted_values'][eng_key]
                if eng_key == "applied_standards":
                    extracted_values_tr[tr_name] = ", ".join(value) if value else "Bulunamadı"
                else:
                    extracted_values_tr[tr_name] = value
        
        response_data = {
            'analysis_date': report.get('analysis_date'),
            'analysis_id': f"at_type_cert_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'filename': filename,
            'file_type': 'AT_TIP_INCELEME_SERTIFIKASI',
            'overall_score': {
                'percentage': round(overall_percentage, 2),
                'total_points': report['summary']['total_score'],
                'max_points': 100,
                'status': status,
                'status_tr': report['summary']['status_tr']
            },
            'category_scores': {},
            'extracted_values': extracted_values_tr,
            'recommendations': report.get('recommendations', []),
            'summary': {
                'is_valid': status == "PASS",
                'conclusion': get_conclusion_message(status, overall_percentage),
                'main_issues': [] if status == "PASS" else get_main_issues(report)
            }
        }
        
        for category, score_data in report['scoring']['category_scores'].items():
            response_data['category_scores'][category] = {
                'score': score_data['normalized'],
                'max_score': score_data['max_weight'],
                'percentage': score_data['percentage'],
                'status': 'PASS' if score_data['percentage'] >= 70 else 'FAIL'
            }
        
        return jsonify({
            'success': True,
            'message': 'AT Type Certificate başarıyla analiz edildi',
            'analysis_service': 'at_type_cert',
            'data': response_data
        })
    
    except Exception as e:
        logger.error(f"API endpoint hatası: {str(e)}")
        return jsonify({'error': 'Server error', 'message': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'service': 'AT Type Certificate Analyzer API',
        'version': '1.0.0'
    })

@app.route('/', methods=['GET'])
def index():
    return jsonify({
        'service': 'AT Type Certificate Analyzer API',
        'version': '1.0.0'
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8015))
    logger.info(f"🚀 AT Type Certificate Analyzer API - Port: {port}")
    app.run(host='0.0.0.0', port=port, debug=False)