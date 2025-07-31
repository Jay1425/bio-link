from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from datetime import datetime
from sqlalchemy.dialects.postgresql import JSON

db = SQLAlchemy()

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    google_id = db.Column(db.String(200), unique=True, nullable=True)
    name = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=True)
    profile_pic_url = db.Column(db.String(200), nullable=True)
    is_verified = db.Column(db.Boolean, default=False)
    role = db.Column(db.String(50), nullable=False, default='patient')
    
    appointments = db.relationship('Appointment', backref='patient', lazy=True) # Changed backref
    reports = db.relationship('Report', backref='user', lazy=True)

# New model for doctors
class Doctor(db.Model, UserMixin):
    __tablename__ = 'doctors'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=True)
    specialization = db.Column(db.String(100), nullable=False, default='General Practice')
    license_number = db.Column(db.String(50), unique=True, nullable=True)
    phone = db.Column(db.String(20), nullable=True)
    location = db.Column(db.String(200), nullable=True)
    profile_pic_url = db.Column(db.String(200), nullable=True)
    is_verified = db.Column(db.Boolean, default=False)
    google_id = db.Column(db.String(255), unique=True, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Legacy fields for backward compatibility
    specialty = db.Column(db.String(120), nullable=True)
    lat = db.Column(db.Float, nullable=True)
    lng = db.Column(db.Float, nullable=True)
    
    # Relationships
    appointments = db.relationship('Appointment', backref='doctor', lazy=True, foreign_keys='Appointment.doctor_id')

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'specialty': self.specialization or self.specialty,
            'specialization': self.specialization or self.specialty,
            'location': self.location or 'Location not specified',
            'phone': self.phone or 'Phone not available',
            'lat': self.lat,
            'lng': self.lng
        }
    
    def get_average_rating(self):
        """Calculate average rating from reviews"""
        if not self.reviews_received:
            return 0.0
        total_rating = sum(review.rating for review in self.reviews_received)
        return round(total_rating / len(self.reviews_received), 1)
    
    def get_review_count(self):
        """Get total number of reviews"""
        return len(self.reviews_received)
    
    def get_rating_distribution(self):
        """Get distribution of ratings (1-5 stars)"""
        distribution = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        for review in self.reviews_received:
            distribution[review.rating] += 1
        return distribution

class Appointment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    appointment_date = db.Column(db.DateTime, nullable=False)
    notes = db.Column(db.Text, nullable=True)
    status = db.Column(db.String(50), nullable=False, default='Pending') # Added status
    
    patient_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    doctor_id = db.Column(db.Integer, db.ForeignKey('doctors.id'), nullable=False)

class Report(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(200), nullable=False)
    upload_date = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)  # Allow NULL for external patients
    notes = db.Column(db.Text, nullable=True)  # Added notes field

    # Enhanced AI prediction fields
    predicted_condition = db.Column(db.String(100), nullable=True)
    confidence_score = db.Column(db.Float, nullable=True)
    status = db.Column(db.String(50), nullable=True)
    prediction_details = db.Column(JSON, nullable=True)
    
    # New fields for ML integration
    file_path = db.Column(db.String(500), nullable=True)
    file_size = db.Column(db.Integer, nullable=True)
    image_type = db.Column(db.String(50), nullable=True)  # brain, lung, skin, etc.
    is_processed = db.Column(db.Boolean, default=False)
    processing_status = db.Column(db.String(50), default='pending')  # pending, processing, completed, failed
    models_used = db.Column(JSON, nullable=True)  # List of models applied
    highest_risk_prediction = db.Column(db.String(100), nullable=True)
    requires_medical_attention = db.Column(db.Boolean, default=False)
    
    # External patient data for non-registered patients
    external_patient_data = db.Column(JSON, nullable=True)  # Store patient info for non-users
    
    def get_latest_prediction(self):
        """Get the most recent ML prediction for this report"""
        return MLPrediction.query.filter_by(report_id=self.id).order_by(MLPrediction.prediction_timestamp.desc()).first()
    
    def get_high_risk_predictions(self):
        """Get all high/critical risk predictions for this report"""
        return MLPrediction.query.filter_by(report_id=self.id).filter(
            MLPrediction.risk_level.in_(['High', 'Critical'])
        ).all()
    
    def to_dict(self):
        latest_prediction = self.get_latest_prediction()
        return {
            'id': self.id,
            'filename': self.filename,
            'upload_date': self.upload_date.isoformat() if self.upload_date else None,
            'predicted_condition': self.predicted_condition,
            'confidence_score': self.confidence_score,
            'status': self.status,
            'is_processed': self.is_processed,
            'processing_status': self.processing_status,
            'requires_medical_attention': self.requires_medical_attention,
            'latest_prediction': latest_prediction.to_dict() if latest_prediction else None
        }

class Chat(db.Model):
    __tablename__ = 'chats'
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    doctor_id = db.Column(db.Integer, db.ForeignKey('doctors.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_message_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_active = db.Column(db.Boolean, default=True)
    
    # Relationships
    patient = db.relationship('User', backref='patient_chats')
    doctor = db.relationship('Doctor', backref='doctor_chats')
    messages = db.relationship('Message', backref='chat', lazy='dynamic', cascade='all, delete-orphan')

    def __repr__(self):
        return f'<Chat {self.id}: Patient {self.patient_id} - Doctor {self.doctor_id}>'

class Message(db.Model):
    __tablename__ = 'messages'
    id = db.Column(db.Integer, primary_key=True)
    chat_id = db.Column(db.Integer, db.ForeignKey('chats.id'), nullable=False)
    sender_type = db.Column(db.String(10), nullable=False)  # 'patient' or 'doctor'
    sender_id = db.Column(db.Integer, nullable=False)  # User ID or Doctor ID
    content = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    is_read = db.Column(db.Boolean, default=False)
    message_type = db.Column(db.String(20), default='text')  # 'text', 'image', 'file', 'video_call'

    def __repr__(self):
        return f'<Message {self.id}: {self.sender_type} - {self.content[:50]}>'

class VideoCall(db.Model):
    __tablename__ = 'video_calls'
    id = db.Column(db.Integer, primary_key=True)
    room_id = db.Column(db.String(100), unique=True, nullable=False)
    patient_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    doctor_id = db.Column(db.Integer, db.ForeignKey('doctors.id'), nullable=False)
    started_at = db.Column(db.DateTime, default=datetime.utcnow)
    ended_at = db.Column(db.DateTime, nullable=True)
    duration_minutes = db.Column(db.Integer, default=0)
    status = db.Column(db.String(20), default='waiting')  # 'waiting', 'active', 'ended'
    
    # Relationships
    patient = db.relationship('User', backref='video_calls_as_patient')
    doctor = db.relationship('Doctor', backref='video_calls_as_doctor')

    def __repr__(self):
        return f'<VideoCall {self.room_id}: Patient {self.patient_id} - Doctor {self.doctor_id}>'

class Review(db.Model):
    __tablename__ = 'reviews'
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    doctor_id = db.Column(db.Integer, db.ForeignKey('doctors.id'), nullable=False)
    appointment_id = db.Column(db.Integer, db.ForeignKey('appointment.id'), nullable=True)
    rating = db.Column(db.Integer, nullable=False)  # 1-5 stars
    comment = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    patient = db.relationship('User', backref='reviews_given')
    doctor = db.relationship('Doctor', backref='reviews_received')
    appointment = db.relationship('Appointment', backref='review')

    def __repr__(self):
        return f'<Review {self.rating} stars for Doctor {self.doctor_id} by Patient {self.patient_id}>'

class MLPrediction(db.Model):
    """Model to store detailed ML prediction results"""
    __tablename__ = 'ml_predictions'
    
    id = db.Column(db.Integer, primary_key=True)
    report_id = db.Column(db.Integer, db.ForeignKey('report.id'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)  # Allow NULL for external patients
    
    # External patient fields
    external_patient_name = db.Column(db.String(100), nullable=True)
    uploaded_by_doctor = db.Column(db.Integer, db.ForeignKey('doctors.id'), nullable=True)
    
    # Model information
    model_name = db.Column(db.String(100), nullable=False)
    model_description = db.Column(db.String(200), nullable=True)
    
    # Prediction results
    predicted_class = db.Column(db.String(100), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    confidence_percentage = db.Column(db.Float, nullable=False)
    
    # Detailed results (JSON)
    class_probabilities = db.Column(JSON, nullable=True)
    all_classes = db.Column(JSON, nullable=True)
    raw_prediction_data = db.Column(JSON, nullable=True)
    
    # Metadata
    prediction_timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    image_path = db.Column(db.String(500), nullable=True)
    processing_time_ms = db.Column(db.Float, nullable=True)
    
    # Risk assessment
    risk_level = db.Column(db.String(20), nullable=True)  # Low, Medium, High, Critical
    requires_doctor_review = db.Column(db.Boolean, default=False)
    
    # Medical Analysis Results
    is_cancer_detected = db.Column(db.Boolean, default=False, nullable=True)
    medical_interpretation = db.Column(db.String(200), nullable=True)
    recommendation = db.Column(db.Text, nullable=True)
    medical_category = db.Column(db.String(100), nullable=True)
    
    # Relationships
    report = db.relationship('Report', backref='ml_predictions')
    user = db.relationship('User', backref='ml_predictions')
    
    def to_dict(self):
        return {
            'id': self.id,
            'model_name': self.model_name,
            'model_description': self.model_description,
            'predicted_class': self.predicted_class,
            'confidence': self.confidence,
            'confidence_percentage': self.confidence_percentage,
            'class_probabilities': self.class_probabilities,
            'all_classes': self.all_classes,
            'risk_level': self.risk_level,
            'requires_doctor_review': self.requires_doctor_review,
            'prediction_timestamp': self.prediction_timestamp.isoformat() if self.prediction_timestamp else None
        }
    
    def get_risk_level(self):
        """Determine risk level based on prediction confidence and class"""
        if self.confidence < 0.5:
            return "Low"
        elif self.confidence < 0.7:
            return "Medium" 
        elif any(keyword in self.predicted_class.lower() for keyword in ['cancer', 'tumor', 'malignant', 'carcinoma']):
            return "Critical"
        else:
            return "High"
    
    def __repr__(self):
        return f'<MLPrediction {self.model_name}: {self.predicted_class} ({self.confidence:.2%})>'