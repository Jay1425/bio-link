import os
import uuid
from werkzeug.utils import secure_filename
import requests
import random
from datetime import datetime, time, timedelta
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from flask import Flask, render_template, redirect, url_for, flash, session, request, jsonify
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from flask_mail import Mail, Message as MailMessage
from google_auth_oauthlib.flow import Flow

from forms import (
    SignupForm, LoginForm, OTPForm, AppointmentForm, 
    UploadReportForm, UpdateProfileForm, ChangePasswordForm,
    DoctorSignupForm, DoctorLoginForm, DoctorUpdateProfileForm, DoctorChangePasswordForm
)
from models import db, User, Appointment, Report, Doctor, Chat, Message, VideoCall, Review, MLPrediction
from flask_migrate import Migrate
from flask_socketio import SocketIO, emit, join_room, leave_room

# Import ML service
try:
    from ml_service import initialize_ml_service, get_ml_service
    ML_ENABLED = True
    print("✅ ML Service loaded successfully")
except ImportError as e:
    ML_ENABLED = False
    print(f"⚠️  ML Service not available: {e}")
    print("   To enable ML features, install: pip install tensorflow pillow opencv-python")

# --- App Initialization and Configuration ---
os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'a-default-secret-key-for-dev-only')
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///biolink.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = os.environ.get('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.environ.get('MAIL_PASSWORD')
app.config['MAIL_DEFAULT_SENDER'] = os.environ.get('MAIL_USERNAME')

# Check if email configuration is complete
if not app.config['MAIL_USERNAME'] or not app.config['MAIL_PASSWORD']:
    print("⚠️  WARNING: Email configuration incomplete!")
    print("   MAIL_USERNAME:", app.config['MAIL_USERNAME'])
    print("   MAIL_PASSWORD:", "***" if app.config['MAIL_PASSWORD'] else None)
    print("   Please set MAIL_USERNAME and MAIL_PASSWORD environment variables for email functionality.")
    print("   See EMAIL_SETUP_GUIDE.md for detailed instructions.")
app.config['Maps_API_KEY'] = os.environ.get('Maps_API_KEY', 'AIzaSyDfiWj8UuBhy8ANJtgMP3UsKGiJFX5OeJU')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['AVATAR_UPLOAD_FOLDER'] = 'static/avatars'
CLIENT_SECRET_FILE = 'client_secret.json'
REDIRECT_URI = 'http://127.0.0.1:5000/callback'  # Changed back to port 5000
SCOPES = ['https://www.googleapis.com/auth/userinfo.profile', 'https://www.googleapis.com/auth/userinfo.email', 'openid']

# --- Initialize Extensions ---
socketio = SocketIO(app, cors_allowed_origins="*")
mail = Mail(app)
db.init_app(app)
migrate = Migrate(app, db)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

# Initialize ML Service
if ML_ENABLED:
    try:
        ml_service = initialize_ml_service()
        print(f"✅ ML Service initialized with {len(ml_service.models)} models")
    except Exception as e:
        print(f"⚠️  ML Service initialization failed: {e}")
        ML_ENABLED = False

with app.app_context():
    db.create_all()

    if not Doctor.query.first():
        # Create sample doctors with the new model structure
        sample_doctors = [
            Doctor(
                name='Dr. Anjali Sharma', 
                specialization='Cardiologist',
                email='anjali.sharma@biolink.com',
                password_hash=generate_password_hash('password123', method='pbkdf2:sha256'),
                license_number='LIC001',
                location='Mumbai, India',
                phone='+91-9876543210',
                is_verified=True
            ),
            Doctor(
                name='Dr. Vikram Singh', 
                specialization='Neurologist',
                email='vikram.singh@biolink.com',
                password_hash=generate_password_hash('password123', method='pbkdf2:sha256'),
                license_number='LIC002',
                location='Delhi, India',
                phone='+91-9876543211',
                is_verified=True
            ),
            Doctor(
                name='Dr. Priya Patel', 
                specialization='Dermatologist',
                email='priya.patel@biolink.com',
                password_hash=generate_password_hash('password123', method='pbkdf2:sha256'),
                license_number='LIC003',
                location='Bangalore, India',
                phone='+91-9876543212',
                is_verified=True
            )
        ]
        db.session.add_all(sample_doctors)
        db.session.commit()

    # Create a sample user for testing if none exists
    if not User.query.first():
        sample_user = User(
            name='Test User',
            email='test@biolink.com',
            password=generate_password_hash('password123', method='pbkdf2:sha256'),
            is_verified=True,
            role='patient'
        )
        db.session.add(sample_user)
        db.session.commit()

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Ensure the avatar upload folder exists
if not os.path.exists(app.config['AVATAR_UPLOAD_FOLDER']):
    os.makedirs(app.config['AVATAR_UPLOAD_FOLDER'])

# --- User Loader and Helper Functions ---
@login_manager.user_loader
def load_user(user_id):
    # Check if it's a patient user
    user = User.query.get(int(user_id))
    if user:
        return user
    # Check if it's a doctor user
    doctor = Doctor.query.get(int(user_id))
    if doctor:
        return doctor
    return None

def send_otp(email, otp):
    # Check if email configuration is available
    if not app.config['MAIL_USERNAME'] or not app.config['MAIL_PASSWORD']:
        print(f"⚠️  Email not configured. OTP for {email}: {otp}")
        flash(f"⚠️ Email not configured. For development, your OTP is: {otp}", "warning")
        return
    
    try:
        msg = MailMessage('Your BioLink Verification Code', recipients=[email])
        msg.body = f'Your One-Time Password (OTP) for account verification is: {otp}'
        mail.send(msg)
        print(f"✅ OTP sent successfully to {email}")
    except Exception as e:
        print(f"❌ Error sending email to {email}: {e}")
        print(f"📧 Development OTP for {email}: {otp}")
        flash(f"Email sending failed. For development, your OTP is: {otp}", "warning")

def get_google_flow():
    return Flow.from_client_secrets_file(
        client_secrets_file=CLIENT_SECRET_FILE,
        scopes=SCOPES,
        redirect_uri=REDIRECT_URI
    )

# --- Authentication Routes ---
@app.route('/')
def index():
    return render_template("index.html")

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    form = SignupForm()
    if form.validate_on_submit():
        if User.query.filter_by(email=form.email.data).first():
            flash("This email address is already registered. Please log in.", "warning")
            return redirect(url_for("login"))
        
        otp = str(random.randint(100000, 999999))
        session['signup_data'] = {
            'name': form.name.data,
            'email': form.email.data,
            'password': generate_password_hash(form.password.data, method='pbkdf2:sha256'),
            'otp': otp
        }
        send_otp(form.email.data, otp)
        flash("An OTP has been sent to your email. Please verify to complete signup.", "info")
        return redirect(url_for("verify"))
    return render_template("signup.html", form=form)

@app.route("/verify", methods=["GET", "POST"])
def verify():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    form = OTPForm()
    signup_data = session.get('signup_data')
    if not signup_data:
        flash("Your session has expired. Please sign up again.", "error")
        return redirect(url_for("signup"))
    if form.validate_on_submit():
        if form.otp.data == signup_data['otp']:
            new_user = User(
                name=signup_data['name'],
                email=signup_data['email'],
                password=signup_data['password'],
                is_verified=True
            )
            db.session.add(new_user)
            db.session.commit()
            session.pop('signup_data', None)
            flash("Account created successfully! Please log in.", "success")
            return redirect(url_for("login"))
        else:
            flash("Invalid OTP. Please try again.", "error")
    # NOTE: You will need to create a 'verify.html' template for this to work.
    return render_template("verify.html", form=form)


@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user and user.password and check_password_hash(user.password, form.password.data):
            if not user.is_verified:
                flash("Your account is not verified. Please complete the OTP verification.", "warning")
                return redirect(url_for("signup"))
            login_user(user)
            flash(f"Welcome back, {user.name}!", "success")
            return redirect(url_for("dashboard"))
        else:
            flash("Invalid email or password. Please try again.", "error")
    return render_template("login.html", form=form)

# --- Google OAuth Routes ---
@app.route("/google/login")
def google_login():
    flow = get_google_flow()
    authorization_url, state = flow.authorization_url()
    session["state"] = state
    return redirect(authorization_url)

@app.route("/callback")
def callback():
    try:
        flow = get_google_flow()
        flow.fetch_token(authorization_response=request.url)
        
        # Check if state exists in session and matches
        if "state" not in session:
            flash("Session expired. Please try logging in again.", "error")
            return redirect(url_for("login"))
            
        if not session.get("state") == request.args.get("state"):
            flash("State mismatch. Possible CSRF attack.", "error")
            return redirect(url_for("login"))
            
        credentials = flow.credentials
        response = requests.get(
            "https://www.googleapis.com/oauth2/v3/userinfo",
            headers={'Authorization': f'Bearer {credentials.token}'}
        )
        user_info = response.json()
        google_id = user_info['sub']
        email = user_info['email']
        name = user_info.get('name', 'User')
        picture = user_info.get('picture')
        user = User.query.filter_by(google_id=google_id).first()
        if not user:
            user = User.query.filter_by(email=email).first()
            if not user:
                user = User(google_id=google_id, name=name, email=email, profile_pic_url=picture, is_verified=True)
                db.session.add(user)
            else:
                user.google_id = google_id
                user.profile_pic_url = picture
            user.is_verified = True
        db.session.commit()
        login_user(user)
        
        # Clear the state from session
        session.pop("state", None)
        
        flash(f"Successfully logged in as {user.name} via Google.", "success")
        return redirect(url_for("dashboard"))
        
    except Exception as e:
        # Log the error and redirect to login
        print(f"OAuth callback error: {e}")
        flash("Login failed. Please try again.", "error")
        return redirect(url_for("login"))

# --- Doctor Authentication Routes ---
@app.route("/doctor/signup", methods=["GET", "POST"])
def doctor_signup():
    if current_user.is_authenticated:
        return redirect(url_for('doctor_dashboard'))
    form = DoctorSignupForm()
    if form.validate_on_submit():
        if Doctor.query.filter_by(email=form.email.data).first():
            flash("This email address is already registered. Please log in.", "warning")
            return redirect(url_for("doctor_login"))
        
        new_doctor = Doctor(
            name=form.name.data,
            email=form.email.data,
            password_hash=generate_password_hash(form.password.data, method='pbkdf2:sha256'),
            specialization=form.specialization.data,
            license_number=form.license_number.data,
            phone=form.phone.data,
            location=form.location.data,
            is_verified=True  # For now, auto-verify doctors
        )
        db.session.add(new_doctor)
        db.session.commit()
        flash("Doctor account created successfully! Please log in.", "success")
        return redirect(url_for("doctor_login"))
    return render_template("doctor_signup.html", form=form)

@app.route("/doctor/login", methods=["GET", "POST"])
def doctor_login():
    if current_user.is_authenticated:
        return redirect(url_for('doctor_dashboard'))
    form = DoctorLoginForm()
    if form.validate_on_submit():
        doctor = Doctor.query.filter_by(email=form.email.data).first()
        if doctor and doctor.password_hash and check_password_hash(doctor.password_hash, form.password.data):
            if not doctor.is_verified:
                flash("Your account is not verified. Please contact administration.", "warning")
                return redirect(url_for("doctor_login"))
            login_user(doctor)
            flash(f"Welcome back, Dr. {doctor.name}!", "success")
            return redirect(url_for("doctor_dashboard"))
        else:
            flash("Invalid email or password. Please try again.", "error")
    return render_template("doctor_login.html", form=form)

@app.route("/doctor/logout")
@login_required
def doctor_logout():
    logout_user()
    flash("You have been successfully logged out.", "success")
    return redirect(url_for("index"))

# --- Application Routes ---

@app.route("/dashboard")
@login_required
def dashboard():
    # Fetch real-time data for the current user from the database
    now = datetime.now()
    
    # Get all appointments
    appointments = Appointment.query.filter_by(patient_id=current_user.id).order_by(Appointment.appointment_date.desc()).all()
    
    # Get upcoming appointments
    upcoming_appointments = Appointment.query.filter(
        Appointment.patient_id == current_user.id,
        Appointment.appointment_date >= now
    ).order_by(Appointment.appointment_date.asc()).limit(5).all()
    
    # Get recent reports with ML predictions
    reports = Report.query.filter_by(user_id=current_user.id).order_by(Report.upload_date.desc()).all()
    recent_reports = reports[:3] if reports else []
    
    # Get all ML predictions for user's reports
    user_predictions = []
    for report in reports:
        ml_predictions = MLPrediction.query.filter_by(report_id=report.id).all()
        for prediction in ml_predictions:
            prediction.report = report  # Add report reference
            user_predictions.append(prediction)
    
    # Sort predictions by date (newest first)
    user_predictions.sort(key=lambda x: x.prediction_timestamp or x.report.upload_date, reverse=True)
    recent_predictions = user_predictions[:5]  # Show latest 5
    
    # Calculate ML-based statistics
    total_ml_analyses = len(user_predictions)
    cancer_detected_count = len([p for p in user_predictions if getattr(p, 'is_cancer_detected', False)])
    high_risk_predictions = len([p for p in user_predictions if p.risk_level in ['High', 'Critical']])
    
    # Get recent chats with doctors
    recent_chats = db.session.query(Chat).filter(
        Chat.patient_id == current_user.id
    ).order_by(Chat.last_message_at.desc()).limit(3).all()
    
    # Calculate live statistics
    total_appointments = len(appointments)
    completed_appointments = len([apt for apt in appointments if apt.appointment_date < now and apt.status == 'Confirmed'])
    pending_appointments = len([apt for apt in appointments if apt.status == 'Pending'])
    
    # Simulated live health data (in real app, this would come from IoT devices)
    import random
    live_heart_rate = random.randint(70, 80)
    live_oxygen = random.randint(96, 99)
    live_temperature = round(random.uniform(98.1, 98.8), 1)
    
    # Real AI results from ML predictions
    ai_results = []
    for prediction in recent_predictions:
        ai_results.append({
            'condition': prediction.predicted_class,
            'confidence': prediction.confidence * 100,
            'risk_level': prediction.risk_level,
            'date': prediction.prediction_timestamp or prediction.report.upload_date,
            'is_cancer_detected': getattr(prediction, 'is_cancer_detected', False),
            'medical_interpretation': getattr(prediction, 'medical_interpretation', prediction.predicted_class),
            'report_id': prediction.report_id
        })
    
    # Graph data for health trends
    graph_data = {
        'dates': [(now - timedelta(days=x)).strftime('%Y-%m-%d') for x in range(7, 0, -1)],
        'heart_rate': [random.randint(70, 80) for _ in range(7)],
        'blood_pressure_sys': [random.randint(110, 130) for _ in range(7)],
        'blood_pressure_dia': [random.randint(70, 85) for _ in range(7)]
    }

    # Pass all data to the template
    return render_template(
        "dashboard.html", 
        appointments=appointments,
        upcoming_appointments=upcoming_appointments,
        reports=reports,
        recent_reports=recent_reports,
        recent_chats=recent_chats,
        ai_results=ai_results,
        live_heart_rate=live_heart_rate,
        live_oxygen=live_oxygen,
        live_temperature=live_temperature,
        total_appointments=total_appointments,
        completed_appointments=completed_appointments,
        pending_appointments=pending_appointments,
        graph_data=graph_data,
        total_ml_analyses=total_ml_analyses,
        cancer_detected_count=cancer_detected_count,
        high_risk_predictions=high_risk_predictions,
        recent_predictions=recent_predictions
    )

@app.route('/ai_prediction')
@login_required
def prediction(): # Renamed to match url_for('prediction')
    # Get all ML predictions for current user
    user_reports = Report.query.filter_by(user_id=current_user.id).all()
    
    # Get all ML predictions for user's reports
    all_predictions = []
    for report in user_reports:
        ml_predictions = MLPrediction.query.filter_by(report_id=report.id).all()
        for prediction in ml_predictions:
            prediction.report = report  # Add report reference
            all_predictions.append(prediction)
    
    # Sort by prediction timestamp (newest first)
    all_predictions.sort(key=lambda x: x.prediction_timestamp or x.report.upload_date, reverse=True)
    
    # Count statistics
    total_predictions = len(all_predictions)
    cancer_detected_count = len([p for p in all_predictions if getattr(p, 'is_cancer_detected', False)])
    high_risk_count = len([p for p in all_predictions if p.risk_level in ['High', 'Critical']])
    
    return render_template("ai_prediction.html", 
                         ml_predictions=all_predictions,
                         total_predictions=total_predictions,
                         cancer_detected_count=cancer_detected_count,
                         high_risk_count=high_risk_count)

# --- Add Placeholder Routes for Sidebar Links ---

# In app.py, replace the entire appointments function with this one

@app.route('/appointments', methods=["GET", "POST"])
@login_required
def appointments():
    # We still use the form object, mainly for the CSRF token
    form = AppointmentForm()
    # The doctor choices are now handled on the frontend with Alpine.js,
    # but we'll fetch the list to pass to the template.
    doctors = Doctor.query.all()

    if request.method == 'POST':
        # Read data directly from the submitted form in the modal
        doctor_id = request.form.get('doctor_id')
        appointment_date_str = request.form.get('date')
        appointment_time_str = request.form.get('time')
        notes = request.form.get('notes', '')

        # Validate that we have the required data
        if not all([doctor_id, appointment_date_str, appointment_time_str]):
            flash('Please select a doctor, date, and time before booking.', 'error')
            return redirect(url_for('appointments'))

        try:
            # Combine date and time strings into a datetime object
            appointment_datetime = datetime.strptime(
                f"{appointment_date_str} {appointment_time_str}",
                '%Y-%m-%d %I:%M %p'
            )
            
            # Check if the appointment is in the future
            if appointment_datetime <= datetime.now():
                flash('Please select a future date and time.', 'error')
                return redirect(url_for('appointments'))

            new_appointment = Appointment(
                patient_id=current_user.id,
                doctor_id=int(doctor_id),
                appointment_date=appointment_datetime,
                notes=notes,
                status='Pending'  # Changed to Pending so doctors can approve
            )
            db.session.add(new_appointment)
            db.session.commit()
            
            print(f"DEBUG: NEW APPOINTMENT CREATED:")
            print(f"  ID: {new_appointment.id}")
            print(f"  Patient: {current_user.name} (ID: {current_user.id})")
            print(f"  Doctor ID: {doctor_id}")
            print(f"  Date: {appointment_datetime}")
            print(f"  Status: {new_appointment.status}")
            
            flash('Your appointment has been booked successfully! Please wait for doctor approval.', 'success')
            return redirect(url_for('appointments'))
            
        except ValueError as e:
            flash('Invalid date or time format. Please try again.', 'error')
            return redirect(url_for('appointments'))
        except Exception as e:
            flash('An error occurred while booking your appointment. Please try again.', 'error')
            return redirect(url_for('appointments'))

    # Logic for GET request (loading the page)
    now = datetime.now()
    upcoming_appointments = Appointment.query.filter(
        Appointment.patient_id == current_user.id,
        Appointment.appointment_date >= now
    ).order_by(Appointment.appointment_date.asc()).all()
    
    past_appointments = Appointment.query.filter(
        Appointment.patient_id == current_user.id,
        Appointment.appointment_date < now
    ).order_by(Appointment.appointment_date.desc()).all()

    return render_template(
        "appointments.html",
        form=form,
        doctors=doctors,
        upcoming_appointments=upcoming_appointments,
        past_appointments=past_appointments,
        now=now
    )

@app.route('/video-call/<int:appointment_id>')
@login_required
def patient_video_call(appointment_id):
    """Start video call for patient appointment"""
    appointment = Appointment.query.get_or_404(appointment_id)
    
    # Verify the appointment belongs to the current user
    if appointment.patient_id != current_user.id:
        flash("Access denied. This appointment doesn't belong to you.", "error")
        return redirect(url_for("appointments"))
    
    # Check if appointment is today or in the future
    if appointment.appointment_date.date() < datetime.now().date():
        flash("Cannot join video call for past appointments.", "error")
        return redirect(url_for("appointments"))
    
    # Generate a unique room ID for the video call
    import uuid
    room_id = f"appointment_{appointment_id}_{uuid.uuid4().hex[:8]}"
    
    return render_template(
        "patient_video_call.html",
        appointment=appointment,
        doctor=appointment.doctor,
        room_id=room_id
    )

@app.route('/review/<int:doctor_id>', methods=['GET', 'POST'])
@login_required
def submit_review(doctor_id):
    """Submit a review for a doctor"""
    doctor = Doctor.query.get_or_404(doctor_id)
    
    # Check if user has had appointments with this doctor
    has_appointment = Appointment.query.filter_by(
        patient_id=current_user.id,
        doctor_id=doctor_id
    ).first()
    
    if not has_appointment:
        flash("You can only review doctors you've had appointments with.", "error")
        return redirect(url_for("appointments"))
    
    # Check if user already reviewed this doctor
    existing_review = Review.query.filter_by(
        patient_id=current_user.id,
        doctor_id=doctor_id
    ).first()
    
    if request.method == 'POST':
        rating = request.form.get('rating', type=int)
        comment = request.form.get('comment', '').strip()
        
        if not rating or rating < 1 or rating > 5:
            flash("Please provide a valid rating (1-5 stars).", "error")
            return redirect(request.url)
        
        if existing_review:
            # Update existing review
            existing_review.rating = rating
            existing_review.comment = comment
            existing_review.created_at = datetime.utcnow()
            flash("Your review has been updated!", "success")
        else:
            # Create new review
            review = Review(
                patient_id=current_user.id,
                doctor_id=doctor_id,
                rating=rating,
                comment=comment
            )
            db.session.add(review)
            flash("Thank you for your review!", "success")
        
        db.session.commit()
        return redirect(url_for("appointments"))
    
    return render_template("submit_review.html", doctor=doctor, existing_review=existing_review)

@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
    form = UploadReportForm()
    
    if form.validate_on_submit():
        try:
            file = form.report_file.data
            image_type = form.image_type.data
            notes = form.notes.data or ""
            
            # Generate secure filename
            filename = secure_filename(file.filename)
            file_ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            unique_filename = f"{current_user.id}_{timestamp}_{filename}"
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            
            # Validate file size again (double check)
            file.seek(0, 2)  # Seek to end to get file size
            file_size = file.tell()
            file.seek(0)  # Reset to beginning
            
            if file_size > 10 * 1024 * 1024:  # 10MB limit
                flash('File size exceeds 10MB limit. Please choose a smaller file.', 'error')
                return render_template('report_upload.html', form=form)
            
            # Save the file
            file.save(save_path)
            
            # Auto-detect image type if not specified properly
            if image_type == 'other' or not image_type:
                filename_lower = filename.lower()
                if any(keyword in filename_lower for keyword in ['brain', 'mri', 'ct', 'neurological']):
                    image_type = 'brain'
                elif any(keyword in filename_lower for keyword in ['lung', 'chest', 'xray', 'x-ray', 'pulmonary']):
                    image_type = 'lung'
                elif any(keyword in filename_lower for keyword in ['skin', 'derma', 'lesion', 'mole']):
                    image_type = 'skin'
                else:
                    image_type = 'other'

            # Create initial report record
            new_report = Report(
                filename=filename,
                user_id=current_user.id,
                file_path=save_path,
                file_size=file_size,
                image_type=image_type,
                is_processed=False,
                processing_status='pending',
                notes=notes
            )
            db.session.add(new_report)
            db.session.commit()

            # Run ML predictions if enabled
            if ML_ENABLED:
                try:
                    # Update status to processing
                    new_report.processing_status = 'processing'
                    db.session.commit()
                    
                    # Get ML service and run predictions
                    ml_service = get_ml_service()
                    
                    # Get recommended models for this image type
                    recommended_models = ml_service.get_model_recommendations(image_type)
                    
                    # Run batch prediction
                    results = ml_service.batch_predict(save_path, recommended_models)
                    
                    # Store detailed ML predictions
                    models_used = []
                    highest_confidence = 0
                    best_prediction = None
                    requires_attention = False
                    cancer_detected = False
                    
                    for model_name, prediction_result in results['predictions'].items():
                        if 'error' not in prediction_result:
                            # Create ML prediction record
                            ml_prediction = MLPrediction(
                                report_id=new_report.id,
                                user_id=current_user.id,
                                model_name=model_name,
                                model_description=prediction_result.get('model_description', ''),
                                predicted_class=prediction_result['predicted_class'],
                                confidence=prediction_result['confidence'],
                                confidence_percentage=prediction_result['confidence_percentage'],
                                class_probabilities=prediction_result.get('class_probabilities', {}),
                                all_classes=prediction_result.get('all_classes', []),
                                raw_prediction_data=prediction_result,
                                image_path=save_path,
                                
                                # Medical analysis fields
                                is_cancer_detected=prediction_result.get('is_cancer_detected', False),
                                medical_interpretation=prediction_result.get('medical_interpretation', ''),
                                recommendation=prediction_result.get('recommendation', ''),
                                medical_category=prediction_result.get('medical_category', '')
                            )
                            
                            # Determine risk level
                            ml_prediction.risk_level = ml_prediction.get_risk_level()
                            ml_prediction.requires_doctor_review = ml_prediction.risk_level in ['High', 'Critical']
                            
                            db.session.add(ml_prediction)
                            models_used.append(model_name)
                            
                            # Track highest confidence prediction
                            if prediction_result['confidence'] > highest_confidence:
                                highest_confidence = prediction_result['confidence']
                                best_prediction = prediction_result
                                
                            # Check if requires medical attention
                            if ml_prediction.requires_doctor_review:
                                requires_attention = True
                            
                            if ml_prediction.is_cancer_detected:
                                cancer_detected = True

                    # Update report with summary results
                    if best_prediction:
                        new_report.predicted_condition = best_prediction['predicted_class']
                        new_report.confidence_score = best_prediction['confidence']
                        new_report.status = 'Analyzed'
                        new_report.highest_risk_prediction = best_prediction['predicted_class']
                        new_report.requires_medical_attention = requires_attention
                        new_report.prediction_details = {
                            "summary": f"Analysis completed using {len(models_used)} AI models",
                            "models_used": models_used,
                            "highest_confidence": highest_confidence,
                            "total_predictions": len(results['predictions']),
                            "cancer_detected": cancer_detected,
                            "processing_time": results.get('processing_time', 'Unknown')
                        }
                        
                        # Generate success message with details
                        success_msg = f'Report "{filename}" analyzed successfully using {len(models_used)} AI models!'
                        if cancer_detected:
                            success_msg += ' ⚠️ Potential concerns detected - please review results carefully.'
                        
                        flash(success_msg, 'success')
                    else:
                        new_report.status = 'Analysis Failed'
                        new_report.prediction_details = {"error": "No successful predictions"}
                        flash(f'Report uploaded but analysis failed. Please try again or contact support.', 'error')

                    new_report.is_processed = True
                    new_report.processing_status = 'completed'
                    new_report.models_used = models_used
                    
                    db.session.commit()
                    
                except Exception as e:
                    # Handle ML processing errors
                    new_report.processing_status = 'failed'
                    new_report.prediction_details = {"error": str(e)}
                    db.session.commit()
                    
                    print(f"ML processing error: {e}")
                    flash(f'Report uploaded but AI analysis failed: {str(e)}', 'warning')
            
            else:
                # Fallback when ML is not enabled
                new_report.predicted_condition = 'Manual Review Required'
                new_report.confidence_score = 0.0
                new_report.status = 'Pending Review'
                new_report.is_processed = True
                new_report.processing_status = 'completed'
                new_report.prediction_details = {
                    "message": "AI analysis not available. Manual review by medical professional required."
                }
                db.session.commit()
                flash(f'Report "{filename}" uploaded successfully. AI analysis unavailable - manual review required.', 'info')

            # Redirect to the AI prediction page to view results
            return redirect(url_for('prediction'))
        
        except Exception as e:
            db.session.rollback()
            print(f"Upload error: {e}")
            flash('An error occurred during upload. Please try again.', 'error')
            return render_template('report_upload.html', form=form)
    
    else:
        # Handle form validation errors
        if form.errors:
            for field, errors in form.errors.items():
                for error in errors:
                    flash(f'{field.replace("_", " ").title()}: {error}', 'error')

    return render_template('report_upload.html', form=form)

# --- ML API Endpoints ---

@app.route('/api/ml/models')
@login_required
def api_ml_models():
    """Get available ML models"""
    if not ML_ENABLED:
        return jsonify({'error': 'ML service not available', 'models': []}), 503
    
    try:
        ml_service = get_ml_service()
        models = ml_service.get_available_models()
        return jsonify({'models': models, 'total': len(models)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/ml/predict/<int:report_id>')
@login_required
def api_ml_predict_report(report_id):
    """Get ML predictions for a specific report"""
    report = Report.query.filter_by(id=report_id, user_id=current_user.id).first_or_404()
    
    predictions = MLPrediction.query.filter_by(report_id=report_id).all()
    
    result = {
        'report_id': report_id,
        'report': report.to_dict(),
        'predictions': [pred.to_dict() for pred in predictions],
        'total_predictions': len(predictions)
    }
    
    return jsonify(result)

@app.route('/api/ml/reanalyze/<int:report_id>', methods=['POST'])
@login_required
def api_ml_reanalyze(report_id):
    """Re-run ML analysis on a specific report"""
    if not ML_ENABLED:
        return jsonify({'error': 'ML service not available'}), 503
    
    report = Report.query.filter_by(id=report_id, user_id=current_user.id).first_or_404()
    
    if not report.file_path or not os.path.exists(report.file_path):
        return jsonify({'error': 'Original file not found'}), 404
    
    try:
        # Clear existing predictions
        MLPrediction.query.filter_by(report_id=report_id).delete()
        
        # Update report status
        report.processing_status = 'processing'
        db.session.commit()
        
        # Get models to use (from request or recommendations)
        models_to_use = request.json.get('models', None) if request.is_json else None
        if not models_to_use:
            ml_service = get_ml_service()
            models_to_use = ml_service.get_model_recommendations(report.image_type)
        
        # Run predictions
        ml_service = get_ml_service()
        results = ml_service.batch_predict(report.file_path, models_to_use)
        
        # Store new predictions
        for model_name, prediction_result in results['predictions'].items():
            if 'error' not in prediction_result:
                ml_prediction = MLPrediction(
                    report_id=report.id,
                    user_id=current_user.id,
                    model_name=model_name,
                    model_description=prediction_result.get('model_description', ''),
                    predicted_class=prediction_result['predicted_class'],
                    confidence=prediction_result['confidence'],
                    confidence_percentage=prediction_result['confidence_percentage'],
                    class_probabilities=prediction_result.get('class_probabilities', {}),
                    all_classes=prediction_result.get('all_classes', []),
                    raw_prediction_data=prediction_result,
                    image_path=report.file_path,
                    
                    # New medical analysis fields
                    is_cancer_detected=prediction_result.get('is_cancer_detected', False),
                    medical_interpretation=prediction_result.get('medical_interpretation', ''),
                    recommendation=prediction_result.get('recommendation', ''),
                    medical_category=prediction_result.get('medical_category', '')
                )
                ml_prediction.risk_level = ml_prediction.get_risk_level()
                ml_prediction.requires_doctor_review = ml_prediction.risk_level in ['High', 'Critical']
                db.session.add(ml_prediction)
        
        # Update report
        report.processing_status = 'completed'
        db.session.commit()
        
        return jsonify({'message': 'Analysis completed', 'report_id': report_id})
        
    except Exception as e:
        report.processing_status = 'failed'
        db.session.commit()
        return jsonify({'error': str(e)}), 500

@app.route('/api/ml/stats')
@login_required
def api_ml_stats():
    """Get ML usage statistics for current user"""
    user_reports = Report.query.filter_by(user_id=current_user.id).all()
    user_predictions = MLPrediction.query.filter_by(user_id=current_user.id).all()
    
    # Calculate statistics
    total_reports = len(user_reports)
    processed_reports = len([r for r in user_reports if r.is_processed])
    high_risk_predictions = len([p for p in user_predictions if p.risk_level in ['High', 'Critical']])
    
    # Model usage statistics
    model_usage = {}
    for prediction in user_predictions:
        model_name = prediction.model_name
        if model_name not in model_usage:
            model_usage[model_name] = 0
        model_usage[model_name] += 1
    
    stats = {
        'total_reports': total_reports,
        'processed_reports': processed_reports,
        'pending_reports': total_reports - processed_reports,
        'total_predictions': len(user_predictions),
        'high_risk_predictions': high_risk_predictions,
        'model_usage': model_usage,
        'requires_medical_attention': len([r for r in user_reports if r.requires_medical_attention])
    }
    
    return jsonify(stats)


# Rename the old prediction route to prediction_result and make it dynamic
@app.route('/prediction/<int:report_id>')
@login_required
def prediction_result(report_id):
    # Fetch the specific report, ensuring it belongs to the current user
    report = Report.query.filter_by(id=report_id, user_id=current_user.id).first_or_404()
    
    # Get all ML predictions for this report
    ml_predictions = MLPrediction.query.filter_by(report_id=report_id).order_by(MLPrediction.confidence.desc()).all()

    # Pass the report data and ML predictions to the new medical analysis template
    return render_template('medical_analysis.html', report=report, ml_predictions=ml_predictions)

# Doctor Dashboard and Routes
@app.route('/doctor/dashboard')
@login_required
def doctor_dashboard():
    # Verify current user is a doctor
    if not isinstance(current_user, Doctor):
        flash("Access denied. Doctor account required.", "error")
        return redirect(url_for("index"))
    
    today_start = datetime.combine(datetime.today(), time.min)
    today_end = datetime.combine(datetime.today(), time.max)
    
    # Get real appointment data
    todays_appointments = Appointment.query.filter(
        Appointment.doctor_id == current_user.id,
        Appointment.appointment_date >= today_start,
        Appointment.appointment_date <= today_end
    ).order_by(Appointment.appointment_date.asc()).all()

    # Get upcoming appointments (next 7 days)
    from datetime import timedelta
    next_week = today_end + timedelta(days=7)
    upcoming_appointments = Appointment.query.filter(
        Appointment.doctor_id == current_user.id,
        Appointment.appointment_date > today_end,
        Appointment.appointment_date <= next_week
    ).order_by(Appointment.appointment_date.asc()).limit(10).all()

    # Get pending appointments for approval
    pending_appointments = Appointment.query.filter(
        Appointment.doctor_id == current_user.id,
        Appointment.status == 'Pending'
    ).order_by(Appointment.appointment_date.asc()).all()

    # Get recent reports from patients (with ML predictions)
    recent_reports = []
    ml_analytics = {
        'total_analyses': 0,
        'cancer_detected': 0,
        'high_risk_cases': 0,
        'pending_review': 0,
        'recent_critical': []
    }
    
    # Get all patient IDs for this doctor
    patient_ids = db.session.query(Appointment.patient_id).filter(
        Appointment.doctor_id == current_user.id
    ).distinct().subquery()
    
    # Get reports from doctor's patients
    patient_reports = db.session.query(Report).filter(
        Report.user_id.in_(patient_ids)
    ).order_by(Report.upload_date.desc()).limit(10).all()
    
    # Get ML predictions for these reports
    for report in patient_reports:
        # Get ML predictions for this report
        ml_predictions = MLPrediction.query.filter_by(report_id=report.id).all()
        
        # Attach ML data to report
        report.ml_predictions = ml_predictions
        report.has_cancer_detected = any(getattr(p, 'is_cancer_detected', False) for p in ml_predictions)
        report.highest_risk = max([p.risk_level for p in ml_predictions if p.risk_level], default='Low')
        report.requires_review = any(getattr(p, 'requires_doctor_review', False) for p in ml_predictions)
        
        recent_reports.append(report)
        
        # Update analytics
        ml_analytics['total_analyses'] += len(ml_predictions)
        if report.has_cancer_detected:
            ml_analytics['cancer_detected'] += 1
        if report.highest_risk in ['High', 'Critical']:
            ml_analytics['high_risk_cases'] += 1
        if report.requires_review:
            ml_analytics['pending_review'] += 1
            
        # Track critical cases
        if report.highest_risk == 'Critical':
            ml_analytics['recent_critical'].append({
                'patient_name': report.user.name if report.user else 'Unknown',
                'report_id': report.id,
                'date': report.upload_date,
                'condition': report.predicted_condition or 'Unknown'
            })
    
    # Limit critical cases to most recent 5
    ml_analytics['recent_critical'] = ml_analytics['recent_critical'][:5]

    # Calculate real statistics (fix the join issue)
    total_patients = db.session.query(User).join(
        Appointment, Appointment.patient_id == User.id
    ).filter(
        Appointment.doctor_id == current_user.id
    ).distinct().count()
    
    total_appointments = Appointment.query.filter_by(doctor_id=current_user.id).count()
    completed_appointments = Appointment.query.filter(
        Appointment.doctor_id == current_user.id,
        Appointment.appointment_date < datetime.now(),
        Appointment.status == 'Confirmed'
    ).count()

    stats = { 
        'new_reports': len(recent_reports),
        'open_slots': max(0, 8 - len(todays_appointments)),  # Assuming 8 slots per day
        'total_patients': total_patients,
        'pending_appointments': len(pending_appointments),
        'total_appointments': total_appointments,
        'completed_appointments': completed_appointments,
        'completion_rate': round((completed_appointments / max(total_appointments, 1)) * 100, 1),
        'ml_analyses': ml_analytics['total_analyses'],
        'cancer_cases': ml_analytics['cancer_detected'],
        'high_risk_cases': ml_analytics['high_risk_cases'],
        'pending_ml_review': ml_analytics['pending_review']
    }

    # Format today's date into a readable string
    today_formatted = datetime.today().strftime('%B %d, %Y')

    return render_template(
        "doctor_dashboard.html", 
        todays_appointments=todays_appointments,
        upcoming_appointments=upcoming_appointments,
        pending_appointments=pending_appointments,
        recent_reports=recent_reports,
        stats=stats,
        today_formatted=today_formatted,
        ml_analytics=ml_analytics
    )

@app.route('/doctor/appointments')
@login_required
def doctor_appointments():
    if not isinstance(current_user, Doctor):
        flash("Access denied. Doctor account required.", "error")
        return redirect(url_for("index"))
    
    # Get all appointments for this doctor
    all_appointments = Appointment.query.filter_by(doctor_id=current_user.id).order_by(Appointment.appointment_date.desc()).all()
    
    # Separate into categories
    now = datetime.now()
    pending_appointments = [apt for apt in all_appointments if apt.status == 'Pending']
    upcoming_appointments = [apt for apt in all_appointments if apt.appointment_date >= now and apt.status == 'Confirmed']
    past_appointments = [apt for apt in all_appointments if apt.appointment_date < now]
    
    return render_template(
        "doctor_appointments.html",
        pending_appointments=pending_appointments,
        upcoming_appointments=upcoming_appointments,
        past_appointments=past_appointments
    )

@app.route('/doctor/patient/<int:patient_id>')
@login_required
def view_patient_profile(patient_id):
    """View detailed patient profile for doctors with comprehensive ML analysis data"""
    if not isinstance(current_user, Doctor):
        flash("Access denied. Doctor account required.", "error")
        return redirect(url_for("index"))
    
    # Get patient info
    patient = User.query.get_or_404(patient_id)
    
    # Verify doctor has had appointments with this patient
    doctor_patient_relationship = Appointment.query.filter(
        Appointment.patient_id == patient_id,
        Appointment.doctor_id == current_user.id
    ).first()
    
    if not doctor_patient_relationship:
        flash("Access denied. You can only view patients you have appointments with.", "error")
        return redirect(url_for("doctor_dashboard"))
    
    # Get appointments history with this doctor
    appointments = Appointment.query.filter(
        Appointment.patient_id == patient_id,
        Appointment.doctor_id == current_user.id
    ).order_by(Appointment.appointment_date.desc()).all()
    
    # Get patient's reports with comprehensive ML analysis
    reports = Report.query.filter_by(user_id=patient_id).order_by(Report.upload_date.desc()).all()
    
    # Enhanced ML analysis data
    patient_ml_data = {
        'total_reports': len(reports),
        'total_analyses': 0,
        'cancer_detected_count': 0,
        'high_risk_count': 0,
        'critical_cases': [],
        'recent_predictions': [],
        'risk_timeline': [],
        'model_insights': {}
    }
    
    # Process each report for ML data
    for report in reports:
        # Get ML predictions for this report
        ml_predictions = MLPrediction.query.filter_by(report_id=report.id).order_by(
            MLPrediction.confidence.desc()
        ).all()
        
        # Attach ML data to report
        report.ml_predictions = ml_predictions
        report.has_cancer_detected = any(getattr(p, 'is_cancer_detected', False) for p in ml_predictions)
        report.highest_risk = max([p.risk_level for p in ml_predictions if p.risk_level], default='Low')
        report.requires_review = any(getattr(p, 'requires_doctor_review', False) for p in ml_predictions)
        report.best_prediction = max(ml_predictions, key=lambda x: x.confidence) if ml_predictions else None
        
        # Update patient analytics
        patient_ml_data['total_analyses'] += len(ml_predictions)
        
        if report.has_cancer_detected:
            patient_ml_data['cancer_detected_count'] += 1
        
        if report.highest_risk in ['High', 'Critical']:
            patient_ml_data['high_risk_count'] += 1
            
        # Track critical cases
        if report.highest_risk == 'Critical':
            patient_ml_data['critical_cases'].append({
                'report_id': report.id,
                'date': report.upload_date,
                'condition': report.predicted_condition or 'Unknown',
                'confidence': report.best_prediction.confidence if report.best_prediction else 0,
                'models_used': len(ml_predictions)
            })
        
        # Add to risk timeline
        if ml_predictions:
            patient_ml_data['risk_timeline'].append({
                'date': report.upload_date,
                'risk_level': report.highest_risk,
                'condition': report.predicted_condition,
                'confidence': report.best_prediction.confidence if report.best_prediction else 0
            })
        
        # Track model usage and insights
        for prediction in ml_predictions:
            model_name = prediction.model_name
            if model_name not in patient_ml_data['model_insights']:
                patient_ml_data['model_insights'][model_name] = {
                    'usage_count': 0,
                    'avg_confidence': 0,
                    'cancer_detections': 0,
                    'high_risk_predictions': 0
                }
            
            insights = patient_ml_data['model_insights'][model_name]
            insights['usage_count'] += 1
            insights['avg_confidence'] = (insights['avg_confidence'] + prediction.confidence) / 2
            
            if getattr(prediction, 'is_cancer_detected', False):
                insights['cancer_detections'] += 1
            if prediction.risk_level in ['High', 'Critical']:
                insights['high_risk_predictions'] += 1
        
        # Add recent predictions (top 10)
        for prediction in ml_predictions[:2]:  # Top 2 per report
            patient_ml_data['recent_predictions'].append({
                'report_id': report.id,
                'model_name': prediction.model_name,
                'predicted_class': prediction.predicted_class,
                'confidence': prediction.confidence,
                'risk_level': prediction.risk_level,
                'is_cancer_detected': getattr(prediction, 'is_cancer_detected', False),
                'medical_interpretation': getattr(prediction, 'medical_interpretation', ''),
                'recommendation': getattr(prediction, 'recommendation', ''),
                'date': prediction.prediction_timestamp or report.upload_date
            })
    
    # Sort and limit data
    patient_ml_data['critical_cases'] = sorted(patient_ml_data['critical_cases'], key=lambda x: x['date'], reverse=True)[:5]
    patient_ml_data['recent_predictions'] = sorted(patient_ml_data['recent_predictions'], key=lambda x: x['date'], reverse=True)[:10]
    patient_ml_data['risk_timeline'] = sorted(patient_ml_data['risk_timeline'], key=lambda x: x['date'], reverse=True)
    
    # Get latest health prediction
    latest_prediction = patient_ml_data['recent_predictions'][0] if patient_ml_data['recent_predictions'] else None
    
    return render_template(
        "doctor_patient_profile.html",
        patient=patient,
        appointments=appointments,
        reports=reports,
        latest_prediction=latest_prediction,
        patient_ml_data=patient_ml_data
    )

@app.route('/doctor/schedule')
@login_required
def doctor_schedule():
    """View doctor's schedule for the week"""
    if not isinstance(current_user, Doctor):
        flash("Access denied. Doctor account required.", "error")
        return redirect(url_for("index"))
    
    # Get appointments for current week + next week (14 days total) - UPDATED FOR 2-WEEK VIEW
    from datetime import timedelta
    today = datetime.today()
    start_of_week = today - timedelta(days=today.weekday())
    end_of_period = start_of_week + timedelta(days=13)  # Show 2 weeks (14 days)
    
    # Debug: Print date range
    print(f"DEBUG: Doctor {current_user.id} checking schedule from {start_of_week} to {end_of_period}")
    
    # Debug: Show ALL appointments for this doctor (not just this week)
    all_appointments = Appointment.query.filter_by(doctor_id=current_user.id).all()
    print(f"DEBUG: Total appointments for doctor {current_user.id}: {len(all_appointments)}")
    for apt in all_appointments:
        print(f"  - {apt.id}: {apt.patient.name} on {apt.appointment_date} ({apt.status})")
    
    week_appointments = Appointment.query.filter(
        Appointment.doctor_id == current_user.id,
        Appointment.appointment_date >= start_of_week,
        Appointment.appointment_date <= end_of_period + timedelta(days=1)  # Include end of day
    ).order_by(Appointment.appointment_date.asc()).all()
    
    # Debug: Print found appointments
    print(f"DEBUG: Found {len(week_appointments)} appointments for doctor {current_user.id}")
    for apt in week_appointments:
        print(f"  - {apt.id}: {apt.patient.name} on {apt.appointment_date} ({apt.status})")
    
    # Group appointments by day
    schedule = {}
    for apt in week_appointments:
        day_key = apt.appointment_date.strftime('%Y-%m-%d')
        if day_key not in schedule:
            schedule[day_key] = []
        schedule[day_key].append(apt)
    
    # Generate 2 weeks of dates (14 days)
    week_dates = []
    for i in range(14):
        date = start_of_week + timedelta(days=i)
        day_key = date.strftime('%Y-%m-%d')
        appointments_for_day = schedule.get(day_key, [])
        print(f"DEBUG: {day_key} has {len(appointments_for_day)} appointments")
        
        week_dates.append({
            'date': date,
            'day_name': date.strftime('%A'),
            'day_key': day_key,
            'appointments': appointments_for_day
        })
    
    # Calculate statistics for the template
    total_appointments = len(week_appointments)
    confirmed_appointments = len([apt for apt in week_appointments if apt.status == 'Confirmed'])
    pending_appointments = len([apt for apt in week_appointments if apt.status == 'Pending'])
    
    print(f"DEBUG: Statistics - Total: {total_appointments}, Confirmed: {confirmed_appointments}, Pending: {pending_appointments}")
    
    return render_template(
        "doctor_schedule.html", 
        week_dates=week_dates, 
        today=today,
        total_appointments=total_appointments,
        confirmed_appointments=confirmed_appointments,
        pending_appointments=pending_appointments
    )

@app.route('/doctor/ml-analytics')
@login_required
def doctor_ml_analytics():
    """Comprehensive ML analytics dashboard for doctors"""
    if not isinstance(current_user, Doctor):
        flash("Access denied. Doctor account required.", "error")
        return redirect(url_for("index"))
    
    # Get all patient IDs for this doctor
    patient_ids = db.session.query(Appointment.patient_id).filter(
        Appointment.doctor_id == current_user.id
    ).distinct().subquery()
    
    # Get all reports from doctor's patients
    all_reports = db.session.query(Report).filter(
        Report.user_id.in_(patient_ids)
    ).order_by(Report.upload_date.desc()).all()
    
    # Get all ML predictions for these reports
    report_ids = [r.id for r in all_reports]
    all_predictions = MLPrediction.query.filter(
        MLPrediction.report_id.in_(report_ids)
    ).order_by(MLPrediction.prediction_timestamp.desc()).all()
    
    # Comprehensive analytics
    analytics = {
        'overview': {
            'total_patients': len(set([r.user_id for r in all_reports])),
            'total_reports': len(all_reports),
            'total_analyses': len(all_predictions),
            'cancer_cases': len([p for p in all_predictions if getattr(p, 'is_cancer_detected', False)]),
            'high_risk_cases': len([p for p in all_predictions if p.risk_level in ['High', 'Critical']]),
            'critical_cases': len([p for p in all_predictions if p.risk_level == 'Critical']),
            'pending_review': len([p for p in all_predictions if getattr(p, 'requires_doctor_review', False)])
        },
        'by_image_type': {},
        'by_risk_level': {'Low': 0, 'Medium': 0, 'High': 0, 'Critical': 0},
        'by_model': {},
        'cancer_detection_by_type': {},
        'confidence_distribution': {'high': 0, 'medium': 0, 'low': 0},
        'recent_critical': [],
        'patient_risk_summary': {},
        'timeline_data': []
    }
    
    # Process predictions for analytics
    for prediction in all_predictions:
        # By image type
        image_type = prediction.report.image_type or 'unknown'
        if image_type not in analytics['by_image_type']:
            analytics['by_image_type'][image_type] = {
                'count': 0, 'cancer_detected': 0, 'high_risk': 0
            }
        analytics['by_image_type'][image_type]['count'] += 1
        
        if getattr(prediction, 'is_cancer_detected', False):
            analytics['by_image_type'][image_type]['cancer_detected'] += 1
            analytics['cancer_detection_by_type'][image_type] = analytics['cancer_detection_by_type'].get(image_type, 0) + 1
        
        if prediction.risk_level in ['High', 'Critical']:
            analytics['by_image_type'][image_type]['high_risk'] += 1
        
        # By risk level
        risk_level = prediction.risk_level or 'Low'
        analytics['by_risk_level'][risk_level] += 1
        
        # By model
        model_name = prediction.model_name
        if model_name not in analytics['by_model']:
            analytics['by_model'][model_name] = {
                'usage_count': 0, 'avg_confidence': 0, 'cancer_detected': 0
            }
        analytics['by_model'][model_name]['usage_count'] += 1
        analytics['by_model'][model_name]['avg_confidence'] += prediction.confidence
        
        if getattr(prediction, 'is_cancer_detected', False):
            analytics['by_model'][model_name]['cancer_detected'] += 1
        
        # Confidence distribution
        if prediction.confidence >= 0.8:
            analytics['confidence_distribution']['high'] += 1
        elif prediction.confidence >= 0.6:
            analytics['confidence_distribution']['medium'] += 1
        else:
            analytics['confidence_distribution']['low'] += 1
        
        # Critical cases
        if prediction.risk_level == 'Critical':
            analytics['recent_critical'].append({
                'patient_name': prediction.user.name if prediction.user else 'Unknown',
                'patient_id': prediction.user_id,
                'report_id': prediction.report_id,
                'condition': prediction.predicted_class,
                'confidence': prediction.confidence,
                'date': prediction.prediction_timestamp or prediction.report.upload_date,
                'model': prediction.model_name
            })
        
        # Patient risk summary
        patient_id = prediction.user_id
        if patient_id not in analytics['patient_risk_summary']:
            analytics['patient_risk_summary'][patient_id] = {
                'name': prediction.user.name if prediction.user else 'Unknown',
                'total_analyses': 0,
                'cancer_detected': False,
                'highest_risk': 'Low',
                'latest_date': None
            }
        
        patient_summary = analytics['patient_risk_summary'][patient_id]
        patient_summary['total_analyses'] += 1
        
        if getattr(prediction, 'is_cancer_detected', False):
            patient_summary['cancer_detected'] = True
        
        # Update highest risk
        current_risk = patient_summary['highest_risk']
        new_risk = prediction.risk_level or 'Low'
        risk_priority = {'Low': 1, 'Medium': 2, 'High': 3, 'Critical': 4}
        if risk_priority.get(new_risk, 1) > risk_priority.get(current_risk, 1):
            patient_summary['highest_risk'] = new_risk
        
        # Update latest date
        pred_date = prediction.prediction_timestamp or prediction.report.upload_date
        if not patient_summary['latest_date'] or pred_date > patient_summary['latest_date']:
            patient_summary['latest_date'] = pred_date
        
        # Timeline data
        analytics['timeline_data'].append({
            'date': pred_date.strftime('%Y-%m-%d'),
            'patient': prediction.user.name if prediction.user else 'Unknown',
            'condition': prediction.predicted_class,
            'risk_level': prediction.risk_level,
            'is_cancer': getattr(prediction, 'is_cancer_detected', False)
        })
    
    # Calculate averages for models
    for model_data in analytics['by_model'].values():
        if model_data['usage_count'] > 0:
            model_data['avg_confidence'] = model_data['avg_confidence'] / model_data['usage_count']
    
    # Sort and limit data
    analytics['recent_critical'] = sorted(analytics['recent_critical'], key=lambda x: x['date'], reverse=True)[:10]
    analytics['timeline_data'] = sorted(analytics['timeline_data'], key=lambda x: x['date'], reverse=True)[:30]
    
    # Convert patient risk summary to list and sort
    analytics['patient_risk_list'] = sorted(
        analytics['patient_risk_summary'].values(),
        key=lambda x: (x['highest_risk'] == 'Critical', x['cancer_detected'], x['latest_date'] or datetime.min),
        reverse=True
    )[:20]
    
    return render_template("doctor_ml_analytics.html", analytics=analytics)

@app.route('/doctor/patient/<int:patient_id>/ml-report')
@login_required  
def doctor_patient_ml_report(patient_id):
    """Detailed ML analysis report for a specific patient"""
    if not isinstance(current_user, Doctor):
        flash("Access denied. Doctor account required.", "error")
        return redirect(url_for("index"))
    
    # Verify doctor-patient relationship
    patient = User.query.get_or_404(patient_id)
    doctor_patient_relationship = Appointment.query.filter(
        Appointment.patient_id == patient_id,
        Appointment.doctor_id == current_user.id
    ).first()
    
    if not doctor_patient_relationship:
        flash("Access denied. You can only view patients you have appointments with.", "error")
        return redirect(url_for("doctor_dashboard"))
    
    # Get all reports and predictions for this patient
    reports = Report.query.filter_by(user_id=patient_id).order_by(Report.upload_date.desc()).all()
    all_predictions = []
    
    for report in reports:
        predictions = MLPrediction.query.filter_by(report_id=report.id).order_by(
            MLPrediction.confidence.desc()
        ).all()
        for pred in predictions:
            pred.report = report
            all_predictions.append(pred)
    
    # Generate comprehensive report
    ml_report = {
        'patient': patient,
        'summary': {
            'total_reports': len(reports),
            'total_predictions': len(all_predictions),
            'cancer_detected': any(getattr(p, 'is_cancer_detected', False) for p in all_predictions),
            'highest_risk_level': max([p.risk_level for p in all_predictions if p.risk_level], default='Low'),
            'models_used': list(set([p.model_name for p in all_predictions])),
            'date_range': {
                'first': min([r.upload_date for r in reports]) if reports else None,
                'latest': max([r.upload_date for r in reports]) if reports else None
            }
        },
        'by_image_type': {},
        'risk_progression': [],
        'model_consensus': {},
        'detailed_predictions': all_predictions[:20],  # Latest 20
        'recommendations': []
    }
    
    # Analysis by image type
    for pred in all_predictions:
        img_type = pred.report.image_type or 'unknown'
        if img_type not in ml_report['by_image_type']:
            ml_report['by_image_type'][img_type] = {
                'count': 0, 'cancer_detected': 0, 'avg_confidence': 0, 'highest_risk': 'Low'
            }
        
        type_data = ml_report['by_image_type'][img_type]
        type_data['count'] += 1
        type_data['avg_confidence'] += pred.confidence
        
        if getattr(pred, 'is_cancer_detected', False):
            type_data['cancer_detected'] += 1
        
        # Update highest risk for this type
        risk_priority = {'Low': 1, 'Medium': 2, 'High': 3, 'Critical': 4}
        if risk_priority.get(pred.risk_level, 1) > risk_priority.get(type_data['highest_risk'], 1):
            type_data['highest_risk'] = pred.risk_level
    
    # Calculate averages
    for type_data in ml_report['by_image_type'].values():
        if type_data['count'] > 0:
            type_data['avg_confidence'] = type_data['avg_confidence'] / type_data['count']
    
    # Risk progression over time
    risk_by_date = {}
    for pred in all_predictions:
        date_key = (pred.prediction_timestamp or pred.report.upload_date).date()
        if date_key not in risk_by_date:
            risk_by_date[date_key] = []
        risk_by_date[date_key].append(pred.risk_level or 'Low')
    
    for date, risks in sorted(risk_by_date.items()):
        # Get highest risk for that date
        risk_priority = {'Low': 1, 'Medium': 2, 'High': 3, 'Critical': 4}
        highest_risk = max(risks, key=lambda x: risk_priority.get(x, 1))
        ml_report['risk_progression'].append({
            'date': date,
            'risk_level': highest_risk,
            'prediction_count': len(risks)
        })
    
    # Model consensus
    for pred in all_predictions:
        model = pred.model_name
        if model not in ml_report['model_consensus']:
            ml_report['model_consensus'][model] = {
                'predictions': [], 'avg_confidence': 0, 'cancer_count': 0
            }
        
        ml_report['model_consensus'][model]['predictions'].append(pred)
        if getattr(pred, 'is_cancer_detected', False):
            ml_report['model_consensus'][model]['cancer_count'] += 1
    
    # Calculate model averages
    for model_data in ml_report['model_consensus'].values():
        if model_data['predictions']:
            model_data['avg_confidence'] = sum(p.confidence for p in model_data['predictions']) / len(model_data['predictions'])
    
    # Generate recommendations
    if ml_report['summary']['cancer_detected']:
        ml_report['recommendations'].append({
            'priority': 'Critical',
            'text': 'Cancer detection flagged by AI. Immediate medical review and follow-up recommended.'
        })
    
    if ml_report['summary']['highest_risk_level'] in ['High', 'Critical']:
        ml_report['recommendations'].append({
            'priority': 'High',
            'text': f'High risk level detected ({ml_report["summary"]["highest_risk_level"]}). Consider additional testing and specialist consultation.'
        })
    
    if len(ml_report['summary']['models_used']) >= 3:
        ml_report['recommendations'].append({
            'priority': 'Medium',
            'text': 'Multiple AI models used for comprehensive analysis. Cross-model validation provides increased diagnostic confidence.'
        })
    
    return render_template("doctor_patient_ml_report.html", ml_report=ml_report)

@app.route('/doctor/chat')
@login_required
def doctor_chat():
    """Doctor chat interface"""
    if not isinstance(current_user, Doctor):
        flash("Access denied. Doctor account required.", "error")
        return redirect(url_for("index"))
    
    # Get recent patients this doctor has appointments with (fix join)
    recent_patients = db.session.query(User).join(
        Appointment, Appointment.patient_id == User.id
    ).filter(
        Appointment.doctor_id == current_user.id
    ).distinct().order_by(Appointment.appointment_date.desc()).limit(20).all()
    
    return render_template("doctor_chat.html", recent_patients=recent_patients)

@app.route('/doctor/video-call/<int:patient_id>')
@login_required
def doctor_video_call(patient_id):
    """Start video call with patient"""
    if not isinstance(current_user, Doctor):
        flash("Access denied. Doctor account required.", "error")
        return redirect(url_for("index"))
    
    patient = User.query.get_or_404(patient_id)
    
    # Generate a unique room ID for the video call
    import uuid
    room_id = str(uuid.uuid4())
    
    return render_template(
        "doctor_video_call.html",
        patient=patient,
        room_id=room_id
    )

@app.route('/doctor/appointment/<int:appointment_id>/approve', methods=['POST'])
@login_required
def approve_appointment(appointment_id):
    if not isinstance(current_user, Doctor):
        flash("Access denied. Doctor account required.", "error")
        return redirect(url_for("index"))
    
    appointment = Appointment.query.filter_by(id=appointment_id, doctor_id=current_user.id).first_or_404()
    appointment.status = 'Confirmed'
    db.session.commit()
    
    # Send real-time notification to patient
    socketio.emit('appointment_status_update', {
        'appointment_id': appointment.id,
        'status': 'Confirmed',
        'message': f'Your appointment with Dr. {current_user.name} has been approved!'
    }, room=f'patient_{appointment.patient_id}')
    
    flash(f"Appointment with {appointment.patient.name} has been approved.", "success")
    return redirect(url_for('doctor_appointments'))

@app.route('/doctor/appointment/<int:appointment_id>/reject', methods=['POST'])
@login_required
def reject_appointment(appointment_id):
    if not isinstance(current_user, Doctor):
        flash("Access denied. Doctor account required.", "error")
        return redirect(url_for("index"))
    
    appointment = Appointment.query.filter_by(id=appointment_id, doctor_id=current_user.id).first_or_404()
    appointment.status = 'Cancelled'
    db.session.commit()
    
    # Send real-time notification to patient
    socketio.emit('appointment_status_update', {
        'appointment_id': appointment.id,
        'status': 'Cancelled',
        'message': f'Your appointment with Dr. {current_user.name} has been cancelled.'
    }, room=f'patient_{appointment.patient_id}')
    
    flash(f"Appointment with {appointment.patient.name} has been cancelled.", "info")
    return redirect(url_for('doctor_appointments'))

@app.route('/doctor/patients')
@login_required
def doctor_patients():
    """Enhanced doctor patients view with comprehensive ML analysis data"""
    if not isinstance(current_user, Doctor):
        flash("Access denied. Doctor account required.", "error")
        return redirect(url_for("index"))
    
    # Get all unique patients who have appointments with this doctor
    patients_with_appointments = db.session.query(User).join(
        Appointment, Appointment.patient_id == User.id
    ).filter(
        Appointment.doctor_id == current_user.id
    ).distinct().all()
    
    # Enhanced patient data with ML analytics
    enhanced_patients = []
    
    for patient in patients_with_appointments:
        # Get appointment statistics
        total_appointments = Appointment.query.filter(
            Appointment.patient_id == patient.id,
            Appointment.doctor_id == current_user.id
        ).count()
        
        latest_appointment = Appointment.query.filter(
            Appointment.patient_id == patient.id,
            Appointment.doctor_id == current_user.id
        ).order_by(Appointment.appointment_date.desc()).first()
        
        # Get ML analysis data
        patient_reports = Report.query.filter_by(user_id=patient.id).all()
        patient_predictions = []
        
        for report in patient_reports:
            predictions = MLPrediction.query.filter_by(report_id=report.id).all()
            for pred in predictions:
                pred.report = report
                patient_predictions.append(pred)
        
        # Calculate ML statistics
        ml_stats = {
            'total_reports': len(patient_reports),
            'total_analyses': len(patient_predictions),
            'cancer_detected': any(getattr(p, 'is_cancer_detected', False) for p in patient_predictions),
            'highest_risk_level': 'Low',
            'latest_analysis_date': None,
            'requires_review': any(getattr(p, 'requires_doctor_review', False) for p in patient_predictions),
            'critical_alerts': 0,
            'avg_confidence': 0
        }
        
        if patient_predictions:
            # Calculate highest risk level
            risk_priority = {'Low': 1, 'Medium': 2, 'High': 3, 'Critical': 4}
            highest_risk = max(patient_predictions, key=lambda x: risk_priority.get(x.risk_level or 'Low', 1))
            ml_stats['highest_risk_level'] = highest_risk.risk_level or 'Low'
            
            # Latest analysis date
            ml_stats['latest_analysis_date'] = max(
                p.prediction_timestamp or p.report.upload_date for p in patient_predictions
            )
            
            # Critical alerts count
            ml_stats['critical_alerts'] = len([p for p in patient_predictions if p.risk_level == 'Critical'])
            
            # Average confidence
            ml_stats['avg_confidence'] = sum(p.confidence for p in patient_predictions) / len(patient_predictions)
        
        # Get recent critical findings
        recent_critical = [
            p for p in patient_predictions 
            if p.risk_level in ['High', 'Critical']
        ]
        recent_critical.sort(key=lambda x: x.prediction_timestamp or x.report.upload_date, reverse=True)
        recent_critical = recent_critical[:3]  # Latest 3 critical findings
        
        # Enhanced patient object
        enhanced_patient = {
            'patient': patient,
            'appointment_stats': {
                'total_appointments': total_appointments,
                'latest_appointment': latest_appointment
            },
            'ml_stats': ml_stats,
            'recent_critical': recent_critical,
            'status_priority': (
                3 if ml_stats['critical_alerts'] > 0 else
                2 if ml_stats['cancer_detected'] else
                1 if ml_stats['requires_review'] else
                0
            )
        }
        
        enhanced_patients.append(enhanced_patient)
    
    # Sort patients by priority (critical cases first)
    enhanced_patients.sort(key=lambda x: (
        x['status_priority'],
        x['ml_stats']['latest_analysis_date'] or datetime.min
    ), reverse=True)
    
    # Summary statistics for the doctor
    summary_stats = {
        'total_patients': len(enhanced_patients),
        'patients_with_cancer': len([p for p in enhanced_patients if p['ml_stats']['cancer_detected']]),
        'patients_critical': len([p for p in enhanced_patients if p['ml_stats']['critical_alerts'] > 0]),
        'patients_pending_review': len([p for p in enhanced_patients if p['ml_stats']['requires_review']]),
        'total_analyses': sum(p['ml_stats']['total_analyses'] for p in enhanced_patients),
        'avg_risk_distribution': {}
    }
    
    # Risk distribution
    risk_counts = {'Low': 0, 'Medium': 0, 'High': 0, 'Critical': 0}
    for patient in enhanced_patients:
        risk_level = patient['ml_stats']['highest_risk_level']
        risk_counts[risk_level] += 1
    summary_stats['avg_risk_distribution'] = risk_counts
    
    return render_template(
        "doctor_patients.html", 
        enhanced_patients=enhanced_patients, 
        summary_stats=summary_stats,
        now=datetime.now()
    )

@app.route('/doctor/high-risk-cases')
@login_required
def doctor_high_risk_cases():
    """View all high-risk and critical ML cases requiring doctor review"""
    if not isinstance(current_user, Doctor):
        flash("Access denied. Doctor account required.", "error")
        return redirect(url_for("index"))
    
    # Get all patient IDs for this doctor
    patient_ids = db.session.query(Appointment.patient_id).filter(
        Appointment.doctor_id == current_user.id
    ).distinct().subquery()
    
    # Get all high-risk and critical predictions
    high_risk_predictions = db.session.query(MLPrediction).join(
        Report, MLPrediction.report_id == Report.id
    ).filter(
        Report.user_id.in_(patient_ids),
        MLPrediction.risk_level.in_(['High', 'Critical'])
    ).order_by(MLPrediction.prediction_timestamp.desc()).all()
    
    # Get predictions requiring doctor review
    review_required_predictions = db.session.query(MLPrediction).join(
        Report, MLPrediction.report_id == Report.id
    ).filter(
        Report.user_id.in_(patient_ids),
        MLPrediction.requires_doctor_review == True
    ).order_by(MLPrediction.prediction_timestamp.desc()).all()
    
    # Combine and deduplicate
    all_critical_cases = list(set(high_risk_predictions + review_required_predictions))
    all_critical_cases.sort(key=lambda x: x.prediction_timestamp or x.report.upload_date, reverse=True)
    
    # Enhanced case data
    enhanced_cases = []
    for prediction in all_critical_cases:
        # Get patient info
        patient = prediction.user
        
        # Get related reports and predictions for context
        related_reports = Report.query.filter_by(user_id=patient.id).count()
        related_predictions = MLPrediction.query.filter_by(user_id=patient.id).count()
        
        # Check if patient has other critical cases
        other_critical = MLPrediction.query.join(Report).filter(
            Report.user_id == patient.id,
            MLPrediction.risk_level.in_(['High', 'Critical']),
            MLPrediction.id != prediction.id
        ).count()
        
        enhanced_case = {
            'prediction': prediction,
            'patient': patient,
            'report': prediction.report,
            'context': {
                'total_reports': related_reports,
                'total_predictions': related_predictions,
                'other_critical_cases': other_critical,
                'days_since_analysis': (datetime.now() - (prediction.prediction_timestamp or prediction.report.upload_date)).days
            },
            'urgency_score': (
                5 if prediction.risk_level == 'Critical' and getattr(prediction, 'is_cancer_detected', False) else
                4 if prediction.risk_level == 'Critical' else
                3 if getattr(prediction, 'is_cancer_detected', False) else
                2 if prediction.risk_level == 'High' else
                1
            )
        }
        enhanced_cases.append(enhanced_case)
    
    # Sort by urgency score and recency
    enhanced_cases.sort(key=lambda x: (
        x['urgency_score'],
        -(x['context']['days_since_analysis'])
    ), reverse=True)
    
    # Statistics
    case_stats = {
        'total_cases': len(enhanced_cases),
        'critical_cases': len([c for c in enhanced_cases if c['prediction'].risk_level == 'Critical']),
        'cancer_cases': len([c for c in enhanced_cases if getattr(c['prediction'], 'is_cancer_detected', False)]),
        'urgent_cases': len([c for c in enhanced_cases if c['urgency_score'] >= 4]),
        'patients_affected': len(set([c['patient'].id for c in enhanced_cases])),
        'by_image_type': {},
        'by_model': {}
    }
    
    # Additional statistics
    for case in enhanced_cases:
        # By image type
        img_type = case['report'].image_type or 'unknown'
        case_stats['by_image_type'][img_type] = case_stats['by_image_type'].get(img_type, 0) + 1
        
        # By model
        model = case['prediction'].model_name
        case_stats['by_model'][model] = case_stats['by_model'].get(model, 0) + 1
    
    return render_template(
        "doctor_high_risk_cases.html",
        enhanced_cases=enhanced_cases,
        case_stats=case_stats
    )

@app.route('/doctor/prediction/<int:prediction_id>/review', methods=['POST'])
@login_required
def doctor_review_prediction(prediction_id):
    """Mark a prediction as reviewed by doctor"""
    if not isinstance(current_user, Doctor):
        flash("Access denied. Doctor account required.", "error")
        return redirect(url_for("index"))
    
    prediction = MLPrediction.query.get_or_404(prediction_id)
    
    # Verify doctor has access to this patient
    doctor_patient_relationship = Appointment.query.filter(
        Appointment.patient_id == prediction.user_id,
        Appointment.doctor_id == current_user.id
    ).first()
    
    if not doctor_patient_relationship:
        flash("Access denied. You can only review patients you have appointments with.", "error")
        return redirect(url_for("doctor_dashboard"))
    
    # Update prediction with doctor review
    review_notes = request.form.get('review_notes', '')
    doctor_assessment = request.form.get('doctor_assessment', '')
    
    # Add doctor review data to prediction
    if not hasattr(prediction, 'doctor_review_data'):
        prediction.doctor_review_data = {}
    
    doctor_review = {
        'reviewed_by': current_user.id,
        'reviewed_at': datetime.now().isoformat(),
        'doctor_name': current_user.name,
        'review_notes': review_notes,
        'doctor_assessment': doctor_assessment,
        'requires_follow_up': request.form.get('requires_follow_up') == 'on'
    }
    
    # Update raw_prediction_data to include doctor review
    if prediction.raw_prediction_data:
        prediction.raw_prediction_data['doctor_review'] = doctor_review
    else:
        prediction.raw_prediction_data = {'doctor_review': doctor_review}
    
    # Mark as no longer requiring review if assessment provided
    if doctor_assessment:
        prediction.requires_doctor_review = False
    
    db.session.commit()
    
    flash(f"Review completed for {prediction.user.name}'s {prediction.model_name} analysis.", "success")
    return redirect(url_for('doctor_high_risk_cases'))

@app.route('/doctor/account')
@login_required
def doctor_account():
    if not isinstance(current_user, Doctor):
        flash("Access denied. Doctor account required.", "error")
        return redirect(url_for("index"))
    
    profile_form = DoctorUpdateProfileForm(obj=current_user)
    password_form = DoctorChangePasswordForm()
    return render_template(
        'doctor_account.html', 
        profile_form=profile_form, 
        password_form=password_form
    )

@app.route('/doctor/account/update', methods=['POST'])
@login_required
def doctor_account_update():
    if not isinstance(current_user, Doctor):
        flash("Access denied. Doctor account required.", "error")
        return redirect(url_for("index"))
    
    form = DoctorUpdateProfileForm()
    if form.validate_on_submit():
        # Update doctor profile
        current_user.name = form.name.data
        current_user.email = form.email.data
        current_user.specialization = form.specialization.data
        current_user.phone = form.phone.data
        current_user.location = form.location.data

        # Handle avatar upload
        if form.avatar.data:
            file = form.avatar.data
            filename = secure_filename(f"doctor_{current_user.id}_{file.filename}")
            save_path = os.path.join(app.config['AVATAR_UPLOAD_FOLDER'], filename)
            file.save(save_path)
            current_user.profile_pic_url = f'avatars/{filename}'

        db.session.commit()
        flash('Your profile has been updated successfully!', 'success')
    else:
        # Handle form errors
        for field, errors in form.errors.items():
            for error in errors:
                flash(f"Error in {getattr(form, field).label.text}: {error}", 'error')

    return redirect(url_for('doctor_account'))

@app.route('/doctor/account/password', methods=['POST'])
@login_required
def doctor_change_password():
    if not isinstance(current_user, Doctor):
        flash("Access denied. Doctor account required.", "error")
        return redirect(url_for("index"))
    
    form = DoctorChangePasswordForm()
    if form.validate_on_submit():
        # Check if the old password is correct
        if check_password_hash(current_user.password_hash, form.old_password.data):
            # Hash and set the new password
            current_user.password_hash = generate_password_hash(form.new_password.data, method='pbkdf2:sha256')
            db.session.commit()
            flash('Your password has been changed successfully!', 'success')
        else:
            flash('Incorrect old password. Please try again.', 'error')
    else:
        for field, errors in form.errors.items():
            for error in errors:
                flash(f"Error in password change: {error}", 'error')

    return redirect(url_for('doctor_account'))

@app.route('/doctor/upload-external', methods=['GET', 'POST'])
@login_required
def doctor_upload_external():
    """Upload medical reports for non-registered patients"""
    if not isinstance(current_user, Doctor):
        flash("Access denied. Doctor account required.", "error")
        return redirect(url_for("index"))
    
    if request.method == 'POST':
        try:
            # Get patient information
            patient_name = request.form.get('patient_name', '').strip()
            patient_age = request.form.get('patient_age', type=int)
            patient_gender = request.form.get('patient_gender', '')
            patient_phone = request.form.get('patient_phone', '').strip()
            patient_email = request.form.get('patient_email', '').strip()
            medical_history = request.form.get('medical_history', '').strip()
            clinical_notes = request.form.get('clinical_notes', '').strip()
            
            # Validate required fields
            if not all([patient_name, patient_age, patient_gender]):
                flash('Patient name, age, and gender are required.', 'error')
                return render_template('doctor_upload_external.html')
            
            # Get file data
            file = request.files.get('report_file')
            image_type = request.form.get('image_type', 'other')
            
            if not file or file.filename == '':
                flash('Please select a medical report file to upload.', 'error')
                return render_template('doctor_upload_external.html')
            
            # Validate file
            filename = secure_filename(file.filename)
            file_ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
            
            allowed_extensions = {'pdf', 'jpg', 'jpeg', 'png', 'tiff', 'bmp'}
            if file_ext not in allowed_extensions:
                flash('Invalid file type. Please upload PDF or image files (JPG, PNG, TIFF, BMP).', 'error')
                return render_template('doctor_upload_external.html')
            
            # Check file size
            file.seek(0, 2)
            file_size = file.tell()
            file.seek(0)
            
            if file_size > 10 * 1024 * 1024:  # 10MB limit
                flash('File size exceeds 10MB limit. Please choose a smaller file.', 'error')
                return render_template('doctor_upload_external.html')
            
            # Generate unique filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            unique_filename = f"ext_patient_{current_user.id}_{timestamp}_{filename}"
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            
            # Save the file
            file.save(save_path)
            
            # Create external patient record in Report with extended patient info
            external_patient_data = {
                'name': patient_name,
                'age': patient_age,
                'gender': patient_gender,
                'phone': patient_phone,
                'email': patient_email,
                'medical_history': medical_history,
                'is_external_patient': True,
                'uploaded_by_doctor': current_user.id,
                'doctor_name': current_user.name
            }
            
            # Create report record for external patient
            new_report = Report(
                filename=filename,
                user_id=None,  # No registered user
                file_path=save_path,
                file_size=file_size,
                image_type=image_type,
                is_processed=False,
                processing_status='pending',
                notes=clinical_notes,
                external_patient_data=external_patient_data
            )
            db.session.add(new_report)
            db.session.commit()
            
            # Run ML predictions if enabled
            if ML_ENABLED:
                try:
                    # Update status to processing
                    new_report.processing_status = 'processing'
                    db.session.commit()
                    
                    # Get ML service and run predictions
                    ml_service = get_ml_service()
                    recommended_models = ml_service.get_model_recommendations(image_type)
                    results = ml_service.batch_predict(save_path, recommended_models)
                    
                    # Store ML predictions for external patient
                    models_used = []
                    highest_confidence = 0
                    best_prediction = None
                    requires_attention = False
                    cancer_detected = False
                    
                    for model_name, prediction_result in results['predictions'].items():
                        if 'error' not in prediction_result:
                            # Create ML prediction record
                            ml_prediction = MLPrediction(
                                report_id=new_report.id,
                                user_id=None,  # No registered user
                                model_name=model_name,
                                model_description=prediction_result.get('model_description', ''),
                                predicted_class=prediction_result['predicted_class'],
                                confidence=prediction_result['confidence'],
                                confidence_percentage=prediction_result['confidence_percentage'],
                                class_probabilities=prediction_result.get('class_probabilities', {}),
                                all_classes=prediction_result.get('all_classes', []),
                                raw_prediction_data=prediction_result,
                                image_path=save_path,
                                
                                # Medical analysis fields
                                is_cancer_detected=prediction_result.get('is_cancer_detected', False),
                                medical_interpretation=prediction_result.get('medical_interpretation', ''),
                                recommendation=prediction_result.get('recommendation', ''),
                                medical_category=prediction_result.get('medical_category', ''),
                                
                                # External patient specific fields
                                external_patient_name=patient_name,
                                uploaded_by_doctor=current_user.id
                            )
                            
                            # Determine risk level
                            ml_prediction.risk_level = ml_prediction.get_risk_level()
                            ml_prediction.requires_doctor_review = ml_prediction.risk_level in ['High', 'Critical']
                            
                            db.session.add(ml_prediction)
                            models_used.append(model_name)
                            
                            # Track highest confidence prediction
                            if prediction_result['confidence'] > highest_confidence:
                                highest_confidence = prediction_result['confidence']
                                best_prediction = prediction_result
                            
                            # Check if requires medical attention
                            if ml_prediction.requires_doctor_review:
                                requires_attention = True
                            
                            if ml_prediction.is_cancer_detected:
                                cancer_detected = True
                    
                    # Update report with summary results
                    if best_prediction:
                        new_report.predicted_condition = best_prediction['predicted_class']
                        new_report.confidence_score = best_prediction['confidence']
                        new_report.status = 'Analyzed'
                        new_report.highest_risk_prediction = best_prediction['predicted_class']
                        new_report.requires_medical_attention = requires_attention
                        new_report.prediction_details = {
                            "summary": f"Analysis completed using {len(models_used)} AI models",
                            "models_used": models_used,
                            "highest_confidence": highest_confidence,
                            "total_predictions": len(results['predictions']),
                            "cancer_detected": cancer_detected,
                            "processing_time": results.get('processing_time', 'Unknown'),
                            "external_patient": True,
                            "analyzed_by_doctor": current_user.name
                        }
                        
                        success_msg = f'External patient report for "{patient_name}" analyzed successfully using {len(models_used)} AI models!'
                        if cancer_detected:
                            success_msg += ' ⚠️ Potential concerns detected - please review results carefully.'
                        
                        flash(success_msg, 'success')
                    else:
                        new_report.status = 'Analysis Failed'
                        new_report.prediction_details = {"error": "No successful predictions", "external_patient": True}
                        flash(f'Report uploaded but analysis failed. Please try again or contact support.', 'error')
                    
                    new_report.is_processed = True
                    new_report.processing_status = 'completed'
                    new_report.models_used = models_used
                    
                    db.session.commit()
                    
                    # Redirect to external patient results view
                    return redirect(url_for('doctor_external_patient_results', report_id=new_report.id))
                    
                except Exception as e:
                    # Handle ML processing errors
                    new_report.processing_status = 'failed'
                    new_report.prediction_details = {"error": str(e), "external_patient": True}
                    db.session.commit()
                    
                    print(f"ML processing error: {e}")
                    flash(f'Report uploaded but AI analysis failed: {str(e)}', 'warning')
            
            else:
                # Fallback when ML is not enabled
                new_report.predicted_condition = 'Manual Review Required'
                new_report.confidence_score = 0.0
                new_report.status = 'Pending Review'
                new_report.is_processed = True
                new_report.processing_status = 'completed'
                new_report.prediction_details = {
                    "message": "AI analysis not available. Manual review by medical professional required.",
                    "external_patient": True
                }
                db.session.commit()
                flash(f'Report for "{patient_name}" uploaded successfully. AI analysis unavailable - manual review required.', 'info')
            
            # Redirect to external patient results view
            return redirect(url_for('doctor_external_patient_results', report_id=new_report.id))
            
        except Exception as e:
            db.session.rollback()
            print(f"External patient upload error: {e}")
            flash('An error occurred during upload. Please try again.', 'error')
            return render_template('doctor_upload_external.html')
    
    # GET request - show upload form
    return render_template('doctor_upload_external.html')

@app.route('/doctor/external-patient-results/<int:report_id>')
@login_required
def doctor_external_patient_results(report_id):
    """View analysis results for external patient report"""
    if not isinstance(current_user, Doctor):
        flash("Access denied. Doctor account required.", "error")
        return redirect(url_for("index"))
    
    # Get the report and verify it belongs to this doctor
    report = Report.query.get_or_404(report_id)
    
    if not report.external_patient_data or report.external_patient_data.get('uploaded_by_doctor') != current_user.id:
        flash("Access denied. You can only view reports you've uploaded.", "error")
        return redirect(url_for("doctor_dashboard"))
    
    # Get ML predictions for this report
    ml_predictions = MLPrediction.query.filter_by(report_id=report_id).order_by(MLPrediction.confidence.desc()).all()
    
    # Get external patient data
    patient_data = report.external_patient_data
    
    # Extract individual fields for template
    patient_name = patient_data.get('name', 'Unknown Patient') if patient_data else 'Unknown Patient'
    patient_age = patient_data.get('age', '') if patient_data else ''
    patient_gender = patient_data.get('gender', '') if patient_data else ''
    patient_phone = patient_data.get('phone', '') if patient_data else ''
    patient_email = patient_data.get('email', '') if patient_data else ''
    medical_history = patient_data.get('medical_history', '') if patient_data else ''
    clinical_notes = report.notes or ''
    image_type = report.image_type or 'other'
    
    return render_template(
        'doctor_external_patient_results.html',
        report=report,
        ml_predictions=ml_predictions,
        patient_data=patient_data,
        patient_name=patient_name,
        patient_age=patient_age,
        patient_gender=patient_gender,
        patient_phone=patient_phone,
        patient_email=patient_email,
        medical_history=medical_history,
        clinical_notes=clinical_notes,
        image_type=image_type
    )

@app.route('/doctor/external-patients')
@login_required
def doctor_external_patients():
    """View all external patients uploaded by this doctor"""
    if not isinstance(current_user, Doctor):
        flash("Access denied. Doctor account required.", "error")
        return redirect(url_for("index"))
    
    # Get all external patient reports uploaded by this doctor
    # SQLite-compatible approach to query JSON field
    all_external_reports = Report.query.filter(
        Report.external_patient_data != None
    ).order_by(Report.upload_date.desc()).all()
    
    # Filter in Python since SQLite JSON operators are limited
    external_reports = []
    for report in all_external_reports:
        if (report.external_patient_data and 
            isinstance(report.external_patient_data, dict) and
            report.external_patient_data.get('uploaded_by_doctor') == current_user.id):
            external_reports.append(report)
    
    # Enhanced external patient data with ML analytics
    enhanced_external_patients = []
    
    for report in external_reports:
        # Get ML predictions for this report
        ml_predictions = MLPrediction.query.filter_by(report_id=report.id).order_by(
            MLPrediction.confidence.desc()
        ).all()
        
        # Attach ML data to report
        report.ml_predictions = ml_predictions
        report.has_cancer_detected = any(getattr(p, 'is_cancer_detected', False) for p in ml_predictions)
        report.highest_risk = max([p.risk_level for p in ml_predictions if p.risk_level], default='Low')
        report.requires_review = any(getattr(p, 'requires_doctor_review', False) for p in ml_predictions)
        report.best_prediction = max(ml_predictions, key=lambda x: x.confidence) if ml_predictions else None
        
        # Get patient data
        patient_data = report.external_patient_data
        
        enhanced_patient = {
            'report': report,
            'patient_data': patient_data,
            'ml_stats': {
                'total_analyses': len(ml_predictions),
                'cancer_detected': report.has_cancer_detected,
                'highest_risk_level': report.highest_risk,
                'requires_review': report.requires_review,
                'avg_confidence': sum(p.confidence for p in ml_predictions) / len(ml_predictions) if ml_predictions else 0,
                'latest_analysis_date': report.upload_date
            },
            'status_priority': (
                3 if report.highest_risk == 'Critical' else
                2 if report.has_cancer_detected else
                1 if report.requires_review else
                0
            )
        }
        
        enhanced_external_patients.append(enhanced_patient)
    
    # Sort by priority and date
    enhanced_external_patients.sort(key=lambda x: (
        x['status_priority'],
        x['ml_stats']['latest_analysis_date']
    ), reverse=True)
    
    # Statistics
    external_stats = {
        'total_external_patients': len(enhanced_external_patients),
        'patients_with_cancer': len([p for p in enhanced_external_patients if p['ml_stats']['cancer_detected']]),
        'patients_critical': len([p for p in enhanced_external_patients if p['ml_stats']['highest_risk_level'] == 'Critical']),
        'patients_pending_review': len([p for p in enhanced_external_patients if p['ml_stats']['requires_review']]),
        'total_analyses': sum(p['ml_stats']['total_analyses'] for p in enhanced_external_patients)
    }
    
    # Calculate additional statistics
    from datetime import timedelta
    today = datetime.now().date()
    week_ago = today - timedelta(days=7)
    month_ago = today - timedelta(days=30)
    
    high_risk_count = len([p for p in enhanced_external_patients if p['ml_stats']['highest_risk_level'] in ['High', 'Critical']])
    recent_uploads_count = len([p for p in enhanced_external_patients if p['report'].upload_date.date() >= week_ago])
    monthly_uploads_count = len([p for p in enhanced_external_patients if p['report'].upload_date.date() >= month_ago])
    
    # Format data for the template
    external_reports_formatted = []
    for patient in enhanced_external_patients:
        patient_data = patient['patient_data']
        report = patient['report']
        ml_stats = patient['ml_stats']
        
        formatted_patient = {
            'id': report.id,
            'name': patient_data.get('name', 'Unknown'),
            'age': patient_data.get('age', ''),
            'gender': patient_data.get('gender', '').title(),
            'phone': patient_data.get('phone', ''),
            'email': patient_data.get('email', ''),
            'upload_date': report.upload_date.strftime('%Y-%m-%d'),
            'diagnosis': report.predicted_condition or 'Pending Analysis',
            'confidence': f"{ml_stats['avg_confidence']:.1f}" if ml_stats['avg_confidence'] > 0 else 'N/A',
            'image_type': (report.image_type or 'Other').title(),
            'risk_level': ml_stats['highest_risk_level']
        }
        external_reports_formatted.append(formatted_patient)
    
    return render_template(
        'doctor_external_patients.html',
        enhanced_external_patients=enhanced_external_patients,
        external_stats=external_stats,
        external_reports=external_reports_formatted,  # For template compatibility
        high_risk_count=high_risk_count,
        recent_uploads_count=recent_uploads_count,
        monthly_uploads_count=monthly_uploads_count
    )

@app.route('/map')
@login_required
def map():
    # Get search parameters
    city = request.args.get('city', '')
    specialty = request.args.get('specialty', '')
    radius = int(request.args.get('radius', 10))
    
    # Fetch all doctors from the database
    doctor_objects = Doctor.query.all()
    
    # Convert the list of Doctor objects into a list of dictionaries with location data
    doctors_list = []
    for doctor in doctor_objects:
        doctor_dict = doctor.to_dict()
        
        # Add sample coordinates (in a real app, this would be stored in database)
        # For demo purposes, adding random locations around default area
        import random
        base_lat = 22.3039
        base_lng = 70.8022
        
        # Generate random coordinates within a reasonable area
        doctor_dict['lat'] = base_lat + random.uniform(-0.1, 0.1)
        doctor_dict['lng'] = base_lng + random.uniform(-0.1, 0.1)
        doctor_dict['distance'] = round(random.uniform(0.5, 15.0), 1)  # km
        doctor_dict['availability'] = random.choice(['Available Now', 'Available Today', 'Next Available: Tomorrow'])
        doctor_dict['rating'] = round(random.uniform(4.2, 5.0), 1)
        doctor_dict['reviews'] = random.randint(50, 200)
        
        # Filter by specialty if provided
        if specialty and doctor_dict['specialty'].lower() != specialty.lower():
            continue
            
        doctors_list.append(doctor_dict)
    
    # Sort by distance
    doctors_list.sort(key=lambda x: x['distance'])
    
    # Get user location (in a real app, this could be from user profile or geolocation)
    user_location = { 
        "lat": 22.3039, 
        "lng": 70.8022,
        "address": "Rajkot, Gujarat, India"
    }
    
    # Get unique specialties for filter dropdown
    specialties = list(set([d['specialty'] for d in doctors_list]))
    specialties.sort()

    return render_template(
        "map.html", 
        doctors=doctors_list,
        user_location=user_location,
        Maps_API_KEY=app.config['Maps_API_KEY'],
        search_city=city,
        search_specialty=specialty,
        search_radius=radius,
        specialties=specialties
    )

@app.route('/account')
@login_required
def account():
    profile_form = UpdateProfileForm(obj=current_user)
    password_form = ChangePasswordForm()
    return render_template(
        'account.html', 
        profile_form=profile_form, 
        password_form=password_form
    )

@app.route('/account/update', methods=['POST'])
@login_required
def account_update():
    form = UpdateProfileForm()
    if form.validate_on_submit():
        # Update username and email
        current_user.name = form.username.data
        current_user.email = form.email.data

        # Handle avatar upload
        if form.avatar.data:
            file = form.avatar.data
            filename = secure_filename(f"{current_user.id}_{file.filename}")
            save_path = os.path.join(app.config['AVATAR_UPLOAD_FOLDER'], filename)
            file.save(save_path)
            current_user.profile_pic_url = f'avatars/{filename}'

        db.session.commit()
        flash('Your profile has been updated successfully!', 'success')
    else:
        # Handle form errors
        for field, errors in form.errors.items():
            for error in errors:
                flash(f"Error in {getattr(form, field).label.text}: {error}", 'error')

    return redirect(url_for('account'))


@app.route('/account/password', methods=['POST'])
@login_required
def change_password():
    form = ChangePasswordForm()
    if form.validate_on_submit():
        # Check if the old password is correct
        if check_password_hash(current_user.password, form.old_password.data):
            # Hash and set the new password
            current_user.password = generate_password_hash(form.new_password.data, method='pbkdf2:sha256')
            db.session.commit()
            flash('Your password has been changed successfully!', 'success')
        else:
            flash('Incorrect old password. Please try again.', 'error')
    else:
        for field, errors in form.errors.items():
            for error in errors:
                flash(f"Error in password change: {error}", 'error')

    return redirect(url_for('account'))

# User-Doctor Contact Feature
@app.route('/contact_doctors')
@login_required
def contact_doctors():
    # Get all active doctors
    doctors = Doctor.query.filter(Doctor.is_verified == True).all()
    
    # Get recent conversations for this user
    recent_chats = db.session.query(Chat).filter(
        Chat.patient_id == current_user.id
    ).order_by(Chat.last_message_at.desc()).limit(5).all()
    
    return render_template(
        'contact_doctors.html', 
        doctors=doctors,
        recent_chats=recent_chats
    )

@app.route('/start_chat/<int:doctor_id>')
@login_required
def start_chat(doctor_id):
    doctor = Doctor.query.get_or_404(doctor_id)
    
    # Check if chat already exists
    existing_chat = Chat.query.filter(
        Chat.patient_id == current_user.id,
        Chat.doctor_id == doctor_id
    ).first()
    
    if not existing_chat:
        # Create new chat
        new_chat = Chat(
            patient_id=current_user.id,
            doctor_id=doctor_id,
            created_at=datetime.utcnow(),
            last_message_at=datetime.utcnow()
        )
        db.session.add(new_chat)
        db.session.commit()
        chat_id = new_chat.id
    else:
        chat_id = existing_chat.id
    
    return redirect(url_for('user_chat', chat_id=chat_id))

@app.route('/user_chat/<int:chat_id>')
@login_required
def user_chat(chat_id):
    chat = Chat.query.get_or_404(chat_id)
    
    # Verify user has access to this chat
    if chat.patient_id != current_user.id:
        flash('Access denied to this chat.', 'error')
        return redirect(url_for('contact_doctors'))
    
    # Get all messages for this chat
    messages = Message.query.filter(
        Message.chat_id == chat_id
    ).order_by(Message.timestamp.asc()).all()
    
    return render_template(
        'user_chat.html',
        chat=chat,
        messages=messages,
        doctor=chat.doctor
    )

# Placeholder for account deletion
@app.route('/account/delete', methods=['POST'])
@login_required
def delete_account():
    # --- Add your account deletion logic here ---
    # Be very careful with this action!
    # Example:
    # db.session.delete(current_user)
    # db.session.commit()
    # logout_user()
    flash('Account deletion feature not yet implemented.', 'info')
    return redirect(url_for('account'))

@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash("You have been successfully logged out.", "success")
    return redirect(url_for("index"))

# --- SocketIO Events ---
@socketio.on('connect')
def handle_connect():
    if current_user.is_authenticated:
        if isinstance(current_user, Doctor):
            socketio.emit('connected', {'message': f'Dr. {current_user.name} connected'})
        else:
            socketio.emit('connected', {'message': f'{current_user.name} connected'})

@socketio.on('disconnect')
def handle_disconnect():
    if current_user.is_authenticated:
        print(f'User {current_user.name} disconnected')

@socketio.on('join_room')
def handle_join_room(data):
    if current_user.is_authenticated:
        room = data.get('room')
        if room:
            join_room(room)
            emit('status', {'message': f'Joined room {room}'})

@socketio.on('patient_emergency')
def handle_patient_emergency(data):
    """Handle emergency alerts from patients"""
    if current_user.is_authenticated:
        # Broadcast emergency alert to all doctors
        socketio.emit('emergency_alert', {
            'patient_id': current_user.id,
            'patient_name': current_user.name,
            'message': data.get('message', 'Emergency assistance needed'),
            'timestamp': datetime.now().isoformat()
        }, broadcast=True, namespace='/doctor')

# Real-time Chat and Video Call Events
@socketio.on('join_chat')
def handle_join_chat(data):
    """Handle doctor-patient chat room joining"""
    if current_user.is_authenticated:
        patient_id = data['patient_id']
        doctor_id = data['doctor_id']
        
        # Create or get existing chat
        chat = Chat.query.filter_by(patient_id=patient_id, doctor_id=doctor_id).first()
        if not chat:
            chat = Chat(patient_id=patient_id, doctor_id=doctor_id)
            db.session.add(chat)
            db.session.commit()
        
        # Join the chat room
        room = f"chat_{chat.id}"
        join_room(room)
        
        emit('chat_joined', {'chat_id': chat.id, 'room': room})

@socketio.on('send_message')
def handle_send_message(data):
    """Handle sending messages in doctor-patient chat"""
    if current_user.is_authenticated:
        patient_id = data['patient_id']
        doctor_id = data['doctor_id']
        message_content = data['message']
        sender_type = data['sender_type']
        
        # Get the chat
        chat = Chat.query.filter_by(patient_id=patient_id, doctor_id=doctor_id).first()
        if chat:
            # Save message to database
            message = Message(
                chat_id=chat.id,
                sender_type=sender_type,
                sender_id=doctor_id if sender_type == 'doctor' else patient_id,
                content=message_content
            )
            db.session.add(message)
            
            # Update chat last message time
            chat.last_message_at = datetime.utcnow()
            db.session.commit()
            
            # Broadcast to chat room
            room = f"chat_{chat.id}"
            emit('receive_message', {
                'message': message_content,
                'sender_type': sender_type,
                'timestamp': message.timestamp.isoformat(),
                'patient_id': patient_id
            }, room=room)

@socketio.on('join_video_call')
def handle_join_video_call(data):
    """Handle video call joining"""
    if current_user.is_authenticated:
        room_id = data['room_id']
        patient_id = data['patient_id']
        doctor_id = data['doctor_id']
        user_type = data['user_type']
        
        # Create or get video call record
        video_call = VideoCall.query.filter_by(room_id=room_id).first()
        if not video_call:
            video_call = VideoCall(
                room_id=room_id,
                patient_id=patient_id,
                doctor_id=doctor_id,
                status='active'
            )
            db.session.add(video_call)
            db.session.commit()
        
        # Join the video call room
        join_room(room_id)
        
        if user_type == 'doctor':
            emit('doctor_joined_call', {'doctor_id': doctor_id}, room=room_id)
        else:
            emit('patient_joined_call', {'patient_id': patient_id}, room=room_id)

@socketio.on('video_call_message')
def handle_video_call_message(data):
    """Handle messages during video calls"""
    if current_user.is_authenticated:
        room_id = data['room_id']
        message = data['message']
        sender_type = data['sender_type']
        sender_name = data['sender_name']
        
        emit('video_call_message_received', {
            'message': message,
            'sender_type': sender_type,
            'sender_name': sender_name
        }, room=room_id)

@socketio.on('end_video_call')
def handle_end_video_call(data):
    """Handle ending video calls"""
    if current_user.is_authenticated:
        room_id = data['room_id']
        
        # Update video call record
        video_call = VideoCall.query.filter_by(room_id=room_id).first()
        if video_call:
            video_call.ended_at = datetime.utcnow()
            video_call.status = 'ended'
            
            # Calculate duration
            duration = video_call.ended_at - video_call.started_at
            video_call.duration_minutes = int(duration.total_seconds() / 60)
            
            db.session.commit()
        
        emit('call_ended', {'room_id': room_id}, room=room_id)

# API Endpoints for Real-time Data
@app.route('/api/doctor/stats')
@login_required
def api_doctor_stats():
    """API endpoint for real-time dashboard stats"""
    if not isinstance(current_user, Doctor):
        return jsonify({'error': 'Access denied'}), 403
    
    today_start = datetime.combine(datetime.today(), time.min)
    today_end = datetime.combine(datetime.today(), time.max)
    
    # Get real-time statistics
    todays_appointments = Appointment.query.filter(
        Appointment.doctor_id == current_user.id,
        Appointment.appointment_date >= today_start,
        Appointment.appointment_date <= today_end
    ).count()

    total_patients = db.session.query(User).join(
        Appointment, Appointment.patient_id == User.id
    ).filter(
        Appointment.doctor_id == current_user.id
    ).distinct().count()
    
    pending_appointments = Appointment.query.filter(
        Appointment.doctor_id == current_user.id,
        Appointment.status == 'Pending'
    ).count()

    return jsonify({
        'todays_appointments': todays_appointments,
        'total_patients': total_patients,
        'pending_appointments': pending_appointments,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/doctor/notifications')
@login_required
def api_doctor_notifications():
    """API endpoint for real-time notifications"""
    if not isinstance(current_user, Doctor):
        return jsonify({'error': 'Access denied'}), 403
    
    # In a real implementation, you'd check for new messages, appointments, etc.
    # For now, return empty notifications
    return jsonify({
        'new_notifications': [],
        'unread_messages': 0,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/doctor/online-patients')
@login_required
def api_online_patients():
    """API endpoint to get online patients for chat"""
    if not isinstance(current_user, Doctor):
        return jsonify({'error': 'Access denied'}), 403
    
    # Get patients who have appointments with this doctor (fix join)
    patients = db.session.query(User).join(
        Appointment, Appointment.patient_id == User.id
    ).filter(
        Appointment.doctor_id == current_user.id
    ).distinct().all()
    
    online_patients = []
    for patient in patients:
        # In a real implementation, you'd check online status
        online_patients.append({
            'id': patient.id,
            'name': patient.name,
            'email': patient.email,
            'is_online': True,  # Mock data
            'last_seen': datetime.now().isoformat()
        })
    
    return jsonify({'patients': online_patients})

@app.route('/api/chat/history/<int:patient_id>')
@login_required
def api_chat_history(patient_id):
    """API endpoint to get chat history with a patient"""
    if not isinstance(current_user, Doctor):
        return jsonify({'error': 'Access denied'}), 403
    
    # Get chat between doctor and patient
    chat = Chat.query.filter_by(
        patient_id=patient_id,
        doctor_id=current_user.id
    ).first()
    
    if not chat:
        return jsonify({'messages': []})
    
    # Get recent messages
    messages = Message.query.filter_by(chat_id=chat.id).order_by(
        Message.timestamp.desc()
    ).limit(50).all()
    
    message_list = []
    for msg in reversed(messages):
        message_list.append({
            'id': msg.id,
            'content': msg.content,
            'sender_type': msg.sender_type,
            'timestamp': msg.timestamp.isoformat(),
            'is_read': msg.is_read
        })
    
    return jsonify({'messages': message_list})

@app.route('/test_schedule/<int:doctor_id>')
def test_schedule(doctor_id):
    """Test endpoint to verify 2-week schedule logic"""
    from datetime import datetime, timedelta
    
    # Set up date calculations (same as doctor_schedule route)
    current_date = datetime.now().date()
    start_of_week = current_date - timedelta(days=current_date.weekday())
    end_of_period = start_of_week + timedelta(days=13)  # 2-week period
    
    # Query appointments
    appointments = Appointment.query.filter(
        Appointment.doctor_id == doctor_id,
        Appointment.appointment_date.between(start_of_week, end_of_period)
    ).all()
    
    # Debug output
    debug_info = {
        'doctor_id': doctor_id,
        'current_date': str(current_date),
        'start_of_week': str(start_of_week),
        'end_of_period': str(end_of_period),
        'total_appointments': len(appointments),
        'appointments': [
            {
                'id': apt.id,
                'date': str(apt.appointment_date),
                'patient': apt.patient.name,
                'status': apt.status
            } for apt in appointments
        ]
    }
    
    return f"<pre>{debug_info}</pre>"

# --- Run Application ---
if __name__ == "__main__":
    # Use port 5000
    import sys
    port = 5000  # Changed back to port 5000
    debug_mode = '--debug' in sys.argv or True  # Enable debug by default
    
    print(f"🚀 Starting BioLink on http://localhost:{port}")
    print(f"📊 ML Status: {len(get_ml_service().models) if ML_ENABLED else 0} models loaded")
    print(f"🔧 Debug Mode: {'ON' if debug_mode else 'OFF'}")
    
    socketio.run(app, debug=debug_mode, host='0.0.0.0', port=port)
