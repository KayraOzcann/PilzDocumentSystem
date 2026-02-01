from models import DocumentType, CriteriaWeight, CriteriaDetail, PatternDefinition, ValidationKeyword, CategoryAction
from database import db

def load_service_config(document_type_code):
    """
    Servise ait tüm verileri veritabanından yükler
    
    Returns:
        {
            'criteria_weights': {},
            'criteria_details': {},
            'pattern_definitions': {},
            'validation_keywords': {},
            'category_actions': {}
        }
    """
    
    # Document type'ı bul
    doc_type = DocumentType.query.filter_by(code=document_type_code, is_active=True).first()
    
    if not doc_type:
        raise ValueError(f"Document type '{document_type_code}' bulunamadı veya aktif değil")
    
    config = {}
    
    # 1. Criteria Weights
    config['criteria_weights'] = {}
    criteria_weights = CriteriaWeight.query.filter_by(
        document_type_id=doc_type.id
    ).order_by(CriteriaWeight.display_order).all()
    
    for cw in criteria_weights:
        config['criteria_weights'][cw.category_name] = cw.weight
    
    # 2. Criteria Details (patternler dahil)
    config['criteria_details'] = {}
    for cw in criteria_weights:
        details = CriteriaDetail.query.filter_by(
            criteria_weight_id=cw.id
        ).order_by(CriteriaDetail.display_order).all()
        
        config['criteria_details'][cw.category_name] = {}
        for detail in details:
            config['criteria_details'][cw.category_name][detail.criterion_name] = {
                'pattern': detail.pattern,
                'weight': detail.weight
            }
    
    # 3. Pattern Definitions (extract_specific_values için)
    config['pattern_definitions'] = {}
    patterns = PatternDefinition.query.filter_by(
        document_type_id=doc_type.id
    ).all()
    
    for pattern in patterns:
        if pattern.pattern_group not in config['pattern_definitions']:
            config['pattern_definitions'][pattern.pattern_group] = {}
        config['pattern_definitions'][pattern.pattern_group][pattern.field_name] = pattern.patterns
    
    # 4. Validation Keywords
    config['validation_keywords'] = {
        'critical_terms': [],
        'strong_keywords': [],
        'excluded_keywords': []
    }
    
    # Critical terms (kategorili)
    critical = ValidationKeyword.query.filter_by(
        document_type_id=doc_type.id,
        keyword_type='critical_terms'
    ).all()
    
    for vk in critical:
        config['validation_keywords']['critical_terms'].append({
            'category': vk.category,
            'keywords': vk.keywords
        })
    
    # Strong keywords
    strong = ValidationKeyword.query.filter_by(
        document_type_id=doc_type.id,
        keyword_type='strong_keywords'
    ).first()
    
    if strong:
        config['validation_keywords']['strong_keywords'] = strong.keywords
    
    # Excluded keywords
    excluded = ValidationKeyword.query.filter_by(
        document_type_id=doc_type.id,
        keyword_type='excluded_keywords'
    ).first()
    
    if excluded:
        config['validation_keywords']['excluded_keywords'] = excluded.keywords
    
    # 5. Category Actions
    config['category_actions'] = {}
    for cw in criteria_weights:
        actions = CategoryAction.query.filter_by(
            criteria_weight_id=cw.id
        ).order_by(CategoryAction.display_order).all()
        
        if actions:
            config['category_actions'][cw.category_name] = [a.action_text for a in actions]
    
    return config