# face_detection
Run the App
Before running the app, create the SQLite database by executing:

from database import create_db
create_db()

Now, you can run your Flask app:

python app.py
Visit http://127.0.0.1:5000/ in your browser.


Steps:
Create a new file create_db.py in your project directory.

This script will call the create_db() function from database.py to set up the database tables.

create_db.py:

from database import create_db

# Call the function to create the database
create_db()

Run the create_db.py script:

In your terminal, run the script to create the database and the necessary table:

python create_db.py
This will initialize the database and the registered_people table.

After running create_db.py, you don't need to execute it again unless you make changes to the database schema. You can now run your app.py to start the web app.

Notes:
You only need to execute create_db() once to set up the database.

Once the database is created, the Flask app (app.py) will interact with it automatically.

So, to summarize:

create_db() goes in database.py.

create_db.py is a one-time script used to initialize the database.

After running create_db.py, you can proceed with running the Flask app (app.py).

PS D:\python projects> .\face\Scripts\activate


(face) PS D:\python projects\face> python app.py
Please install face_recognition_models with this command before using face_recognition:

pip install git+https://github.com/ageitgey/face_recognition_models

pip install git+https://github.com/ageitgey/face_recognition_models

python app.py
