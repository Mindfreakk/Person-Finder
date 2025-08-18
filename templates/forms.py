from flask_wtf import FlaskForm
from wtforms import StringField, IntegerField, TextAreaField, DateField, SubmitField
from wtforms.validators import DataRequired, Length, NumberRange, Regexp
from flask_wtf.file import FileField, FileAllowed

# ----------------- Register Form -----------------
class RegisterForm(FlaskForm):
    full_name = StringField(
        "Full Name",
        validators=[DataRequired(), Length(min=2, max=100)]
    )

    guardian_name = StringField(
        "Guardian Name",
        validators=[DataRequired(), Length(min=2, max=100)]
    )

    age = IntegerField(
        "Age",
        validators=[DataRequired(), NumberRange(min=0, max=120)]
    )

    phone_number = StringField(
        "Phone Number",
        validators=[
            DataRequired(),
            Regexp(r'^[0-9]{10,15}$', message="Enter a valid phone number")
        ]
    )

    address = TextAreaField(
        "Address",
        validators=[DataRequired(), Length(min=5, max=250)]
    )

    last_seen_date = DateField(
        "Last Seen Date",
        format="%Y-%m-%d",
        validators=[DataRequired()],
        render_kw={"placeholder": "YYYY-MM-DD"}
    )

    photo = FileField(
        "Upload Photo",
        validators=[FileAllowed(['jpg', 'jpeg', 'png'], "Only JPG and PNG images are allowed!")]
    )

    submit = SubmitField("Register")


# ----------------- Search Form -----------------
class SearchForm(FlaskForm):
    last_seen = StringField(
        "Last Seen",
        render_kw={"placeholder": "City, YYYY-MM-DD"}
    )
    photo = FileField(
        "Upload Photo",
        validators=[FileAllowed(['jpg', 'jpeg', 'png'], "Only JPG and PNG images are allowed!")]
    )
    submit = SubmitField("Search")

# forms.py
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import DataRequired, Email, Length


class RegisterForm(FlaskForm):
    """Form for registering a new user"""
    username = StringField(
        "Username",
        validators=[DataRequired(), Length(min=3, max=25)]
    )
    email = StringField(
        "Email",
        validators=[DataRequired(), Email()]
    )
    password = PasswordField(
        "Password",
        validators=[DataRequired(), Length(min=6)]
    )
    submit = SubmitField("Register")


class SearchForm(FlaskForm):
    """Form for searching a missing person"""
    query = StringField(
        "Enter name or details",
        validators=[DataRequired(), Length(min=2)]
    )
    submit = SubmitField("Search")
