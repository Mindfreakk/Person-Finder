from flask_wtf import FlaskForm
from wtforms import StringField, IntegerField, TextAreaField, DateField, SubmitField
from wtforms.validators import DataRequired, Length, NumberRange, Regexp

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

    submit = SubmitField("Register")