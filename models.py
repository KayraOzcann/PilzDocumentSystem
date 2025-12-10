from database import db
from datetime import datetime
from sqlalchemy.dialects.postgresql import JSONB

# ============================================
# 1. DOCUMENT TYPES
# ============================================
class DocumentType(db.Model):
    __tablename__ = 'document_types'
    
    id = db.Column(db.Integer, primary_key=True)
    code = db.Column(db.String(50), unique=True, nullable=False, index=True)
    name = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text)
    service_file = db.Column(db.String(100))
    endpoint = db.Column(db.String(100))
    icon = db.Column(db.String(50))
    app_variable_name = db.Column(db.String(100))  
    is_active = db.Column(db.Boolean, default=True, index=True)
    needs_ocr = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.now)
    updated_at = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now)
    
    # Relationships
    criteria_weights = db.relationship('CriteriaWeight', backref='document_type', cascade='all, delete-orphan', lazy='dynamic')
    pattern_definitions = db.relationship('PatternDefinition', backref='document_type', cascade='all, delete-orphan', lazy='dynamic')
    validation_keywords = db.relationship('ValidationKeyword', backref='document_type', cascade='all, delete-orphan', lazy='dynamic')
    
    def __repr__(self):
        return f'<DocumentType {self.code}>'


# ============================================
# 2. CRITERIA WEIGHTS
# ============================================
class CriteriaWeight(db.Model):
    __tablename__ = 'criteria_weights'
    
    id = db.Column(db.Integer, primary_key=True)
    document_type_id = db.Column(db.Integer, db.ForeignKey('document_types.id', ondelete='CASCADE'), nullable=False, index=True)
    category_name = db.Column(db.String(200), nullable=False)
    weight = db.Column(db.Integer, nullable=False)
    display_order = db.Column(db.Integer, default=0)
    created_at = db.Column(db.DateTime, default=datetime.now)
    
    # Relationships
    criteria_details = db.relationship('CriteriaDetail', backref='criteria_weight', cascade='all, delete-orphan', lazy='dynamic')
    category_actions = db.relationship('CategoryAction', backref='criteria_weight', cascade='all, delete-orphan', lazy='dynamic')
    
    __table_args__ = (
        db.UniqueConstraint('document_type_id', 'category_name', name='uq_doc_category'),
    )
    
    def __repr__(self):
        return f'<CriteriaWeight {self.category_name}>'


# ============================================
# 3. CRITERIA DETAILS
# ============================================
class CriteriaDetail(db.Model):
    __tablename__ = 'criteria_details'
    
    id = db.Column(db.Integer, primary_key=True)
    criteria_weight_id = db.Column(db.Integer, db.ForeignKey('criteria_weights.id', ondelete='CASCADE'), nullable=False, index=True)
    criterion_name = db.Column(db.String(200), nullable=False)
    pattern = db.Column(db.Text, nullable=False)
    weight = db.Column(db.Integer, nullable=False)
    display_order = db.Column(db.Integer, default=0)
    created_at = db.Column(db.DateTime, default=datetime.now)
    
    def __repr__(self):
        return f'<CriteriaDetail {self.criterion_name}>'


# ============================================
# 4. PATTERN DEFINITIONS
# ============================================
class PatternDefinition(db.Model):
    __tablename__ = 'pattern_definitions'
    
    id = db.Column(db.Integer, primary_key=True)
    document_type_id = db.Column(db.Integer, db.ForeignKey('document_types.id', ondelete='CASCADE'), nullable=False, index=True)
    pattern_group = db.Column(db.String(100), nullable=False, index=True)
    field_name = db.Column(db.String(100), nullable=False)
    patterns = db.Column(JSONB, nullable=False)
    display_order = db.Column(db.Integer, default=0)
    created_at = db.Column(db.DateTime, default=datetime.now)
    
    def __repr__(self):
        return f'<PatternDefinition {self.field_name}>'


# ============================================
# 5. VALIDATION KEYWORDS
# ============================================
class ValidationKeyword(db.Model):
    __tablename__ = 'validation_keywords'
    
    id = db.Column(db.Integer, primary_key=True)
    document_type_id = db.Column(db.Integer, db.ForeignKey('document_types.id', ondelete='CASCADE'), nullable=False, index=True)
    keyword_type = db.Column(db.String(50), nullable=False, index=True)
    category = db.Column(db.String(100))
    keywords = db.Column(JSONB, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.now)
    
    def __repr__(self):
        return f'<ValidationKeyword {self.keyword_type}>'


# ============================================
# 6. CATEGORY ACTIONS
# ============================================
class CategoryAction(db.Model):
    __tablename__ = 'category_actions'
    
    id = db.Column(db.Integer, primary_key=True)
    criteria_weight_id = db.Column(db.Integer, db.ForeignKey('criteria_weights.id', ondelete='CASCADE'), nullable=False, index=True)
    action_text = db.Column(db.Text, nullable=False)
    display_order = db.Column(db.Integer, default=0)
    created_at = db.Column(db.DateTime, default=datetime.now)
    
    def __repr__(self):
        return f'<CategoryAction {self.id}>'


# ============================================
# 7. TICKETS
# ============================================
class Ticket(db.Model):
    __tablename__ = 'tickets'
    
    id = db.Column(db.Integer, primary_key=True)
    ticket_no = db.Column(db.String(50), unique=True, nullable=False, index=True)
    ticket_id = db.Column(db.String(100), unique=True, nullable=False, index=True)
    document_name = db.Column(db.String(255), nullable=False)
    document_type = db.Column(db.String(100))
    document_url = db.Column(db.Text)
    opening_date = db.Column(db.DateTime, nullable=False, index=True)
    last_updated = db.Column(db.DateTime)
    closing_date = db.Column(db.DateTime)
    status = db.Column(db.String(50), default='Kapalı', index=True)
    responsible = db.Column(db.String(100), default='Savaş Bey')
    analysis_result = db.Column(JSONB)
    inspector_comment = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.now)
    
    # Relationships
    comments = db.relationship('TicketComment', backref='ticket', cascade='all, delete-orphan', lazy='dynamic')
    
    def __repr__(self):
        return f'<Ticket {self.ticket_no}>'


# ============================================
# 8. TICKET COMMENTS
# ============================================
class TicketComment(db.Model):
    __tablename__ = 'ticket_comments'
    
    id = db.Column(db.Integer, primary_key=True)
    ticket_id = db.Column(db.Integer, db.ForeignKey('tickets.id', ondelete='CASCADE'), nullable=False, index=True)
    comment_id = db.Column(db.String(50), nullable=False)
    author = db.Column(db.String(200), nullable=False)
    text = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.now, index=True)
    
    def __repr__(self):
        return f'<TicketComment {self.comment_id}>'


# ============================================
# 9. ANALYSIS EVALUATIONS
# ============================================
class AnalysisEvaluation(db.Model):
    __tablename__ = 'analysis_evaluations'
    
    id = db.Column(db.Integer, primary_key=True)
    analysis_id = db.Column(db.String(100), unique=True, nullable=False, index=True)
    timestamp = db.Column(db.DateTime, nullable=False, index=True)
    analysis_date = db.Column(db.DateTime)
    document_name = db.Column(db.String(255))
    document_extension = db.Column(db.String(10))
    document_size = db.Column(db.String(50))
    document_size_bytes = db.Column(db.BigInteger)
    report_type = db.Column(db.String(100), index=True)
    report_type_display = db.Column(db.String(200))
    overall_score = db.Column(JSONB)
    evaluation_notes = db.Column(db.Text)
    analysis_data = db.Column(JSONB)
    created_at = db.Column(db.DateTime, default=datetime.now)
    
    def __repr__(self):
        return f'<AnalysisEvaluation {self.analysis_id}>'