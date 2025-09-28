# forms.py
from flask_wtf import FlaskForm
from wtforms import StringField, IntegerField, SubmitField, FileField
from wtforms.validators import DataRequired, Length, Optional

print("✅ forms.py loaded successfully")

class RegisterForm(FlaskForm):
    full_name = StringField("Full Name", validators=[DataRequired(), Length(min=2, max=100)])
    guardian_name = StringField("Guardian Name", validators=[Optional(), Length(max=100)])
    age = IntegerField("Age", validators=[Optional()])
    phone = StringField("Phone", validators=[Optional(), Length(max=15)])
    address = StringField("Address", validators=[Optional(), Length(max=200)])
    last_seen = StringField("Last Seen Location", validators=[Optional(), Length(max=200)])
    photo = FileField("Photo", validators=[Optional()])
    submit = SubmitField("Register Person")

class SearchForm(FlaskForm):
    last_seen = StringField("Last Seen Location", validators=[Optional(), Length(max=200)])
    photo = FileField("Upload Photo", validators=[Optional()])
    submit = SubmitField("Search")
