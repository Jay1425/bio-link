from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms import StringField, PasswordField, SubmitField, SelectField, TextAreaField
from wtforms.fields import DateField, TimeField
from wtforms.validators import InputRequired, Email, Length, EqualTo, Optional, ValidationError

class SignupForm(FlaskForm):
    name = StringField("Name", validators=[InputRequired(), Length(min=2, max=100)])
    email = StringField("Email", validators=[InputRequired(), Email(), Length(max=150)])
    password = PasswordField("Password", validators=[InputRequired(), Length(min=6)])
    confirm_password = PasswordField("Confirm Password", validators=[InputRequired(), EqualTo("password")])
    submit = SubmitField("Sign Up")

class LoginForm(FlaskForm):
    email = StringField("Email", validators=[InputRequired(), Email(), Length(max=150)])
    password = PasswordField("Password", validators=[InputRequired()])
    submit = SubmitField("Log In")

class OTPForm(FlaskForm):
    otp = StringField("Enter OTP", validators=[InputRequired(), Length(min=6, max=6)])
    submit = SubmitField("Verify OTP")

# Add this new form for booking appointments
class AppointmentForm(FlaskForm):
    doctor = SelectField("Select a Doctor", coerce=int, validators=[InputRequired()])
    date = DateField("Date", validators=[InputRequired()], format='%Y-%m-%d')
    time = TimeField("Time", validators=[InputRequired()], format='%H:%M')
    notes = TextAreaField("Reason for Visit (Optional)")
    submit = SubmitField("Request Appointment")


class UploadReportForm(FlaskForm):
    report_file = FileField(
        'Medical Report', 
        validators=[
            FileRequired(message="Please select a file to upload."),
            FileAllowed(['pdf', 'png', 'jpg', 'jpeg', 'tiff', 'bmp'], 
                       'Only PDF and image files (JPG, PNG, TIFF, BMP) are allowed!')
        ]
    )
    image_type = SelectField(
        'Medical Image Type',
        choices=[
            ('brain', 'Brain/Neurological (MRI, CT)'),
            ('lung', 'Lung/Chest (X-ray, CT)'),
            ('skin', 'Skin/Dermatological'),
            ('other', 'Other/General')
        ],
        validators=[InputRequired(message="Please select the type of medical image.")],
        default='brain'
    )
    notes = TextAreaField(
        'Additional Notes',
        validators=[Optional()],
        render_kw={"placeholder": "Any additional information about the medical image or symptoms..."}
    )
    submit = SubmitField('Analyze Report')
    
    def validate_report_file(self, field):
        """Custom validation for file size and type"""
        if field.data:
            # Check file size (10MB limit)
            field.data.seek(0, 2)  # Seek to end of file
            file_size = field.data.tell()
            field.data.seek(0)  # Reset to beginning
            
            if file_size > 10 * 1024 * 1024:  # 10MB in bytes
                raise ValidationError('File size cannot exceed 10MB.')
            
            # Additional validation for image files
            if field.data.content_type and field.data.content_type.startswith('image/'):
                # You could add additional image validation here
                pass

class UpdateProfileForm(FlaskForm):
    username = StringField("Username", validators=[InputRequired(), Length(min=2, max=100)])
    email = StringField("Email", validators=[InputRequired(), Email(), Length(max=150)])
    avatar = FileField(
        'Avatar', 
        validators=[
            Optional(), 
            FileAllowed(['jpg', 'jpeg', 'png'], 'Only image files are allowed!')
        ]
    )
    submit = SubmitField('Save Changes')

class ChangePasswordForm(FlaskForm):
    old_password = PasswordField("Old Password", validators=[InputRequired()])
    new_password = PasswordField("New Password", validators=[InputRequired(), Length(min=6)])
    confirm_password = PasswordField(
        "Confirm New Password", 
        validators=[InputRequired(), EqualTo("new_password", message="Passwords must match.")]
    )
    submit = SubmitField('Update Password')

# Doctor Forms
class DoctorSignupForm(FlaskForm):
    name = StringField('Full Name', validators=[InputRequired(), Length(min=2, max=100)])
    email = StringField('Email', validators=[InputRequired(), Email()])
    password = PasswordField('Password', validators=[InputRequired(), Length(min=6)])
    confirm_password = PasswordField('Confirm Password', 
                                   validators=[InputRequired(), EqualTo('password')])
    specialization = StringField('Specialization', validators=[InputRequired(), Length(min=2, max=100)])
    license_number = StringField('Medical License Number', validators=[InputRequired(), Length(min=5, max=50)])
    phone = StringField('Phone Number', validators=[Optional(), Length(max=20)])
    location = StringField('Practice Location', validators=[Optional(), Length(max=200)])
    submit = SubmitField('Create Doctor Account')

class DoctorLoginForm(FlaskForm):
    email = StringField('Email', validators=[InputRequired(), Email()])
    password = PasswordField('Password', validators=[InputRequired()])
    submit = SubmitField('Login')

class DoctorUpdateProfileForm(FlaskForm):
    name = StringField('Full Name', validators=[InputRequired(), Length(min=2, max=100)])
    email = StringField('Email', validators=[InputRequired(), Email()])
    specialization = StringField('Specialization', validators=[InputRequired(), Length(min=2, max=100)])
    phone = StringField('Phone Number', validators=[Optional(), Length(max=20)])
    location = StringField('Practice Location', validators=[Optional(), Length(max=200)])
    avatar = FileField('Profile Picture', validators=[Optional(), FileAllowed(['jpg', 'png', 'jpeg'], 'Images only!')])
    submit = SubmitField('Update Profile')

class DoctorChangePasswordForm(FlaskForm):
    old_password = PasswordField('Current Password', validators=[InputRequired()])
    new_password = PasswordField('New Password', validators=[InputRequired(), Length(min=6)])
    confirm_password = PasswordField('Confirm New Password', 
                                   validators=[InputRequired(), EqualTo('new_password')])
    submit = SubmitField('Change Password')

class DoctorExternalPatientUploadForm(FlaskForm):
    # Patient Information
    patient_name = StringField('Patient Full Name', validators=[InputRequired(), Length(min=2, max=100)])
    patient_age = StringField('Age', validators=[InputRequired()])
    patient_gender = SelectField('Gender', 
                               choices=[('male', 'Male'), ('female', 'Female'), ('other', 'Other')],
                               validators=[InputRequired()])
    patient_phone = StringField('Phone Number', validators=[Optional(), Length(max=20)])
    patient_email = StringField('Email Address', validators=[Optional(), Email(), Length(max=150)])
    
    # Medical Information
    medical_history = TextAreaField('Medical History', validators=[Optional()],
                                  render_kw={"placeholder": "Previous medical conditions, surgeries, medications..."})
    clinical_notes = TextAreaField('Clinical Notes', validators=[Optional()],
                                 render_kw={"placeholder": "Current symptoms, examination findings, clinical observations..."})
    
    # File Upload
    report_file = FileField(
        'Medical Report', 
        validators=[
            FileRequired(message="Please select a medical report file to upload."),
            FileAllowed(['pdf', 'png', 'jpg', 'jpeg', 'tiff', 'bmp'], 
                       'Only PDF and image files (JPG, PNG, TIFF, BMP) are allowed!')
        ]
    )
    image_type = SelectField(
        'Medical Image Type',
        choices=[
            ('brain', 'Brain/Neurological (MRI, CT)'),
            ('lung', 'Lung/Chest (X-ray, CT)'),
            ('skin', 'Skin/Dermatological'),
            ('other', 'Other/General')
        ],
        validators=[InputRequired(message="Please select the type of medical image.")],
        default='other'
    )
    
    submit = SubmitField('Upload and Analyze Report')