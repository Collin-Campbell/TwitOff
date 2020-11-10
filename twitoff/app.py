"""Main app/routing file for Twitoff"""

from flask import Flask, render_template
from .models import DB, User, insert_example_users


# creates application
def create_app():
    """Creating and configuring an instance of the Flask application"""
    app = Flask(__name__)

    # database and app configurations
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///db.sqlite3"
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

    # initilizing database
    DB.init_app(app)

    # decorator listens for specific endpoint visits
    @app.route('/')  # http://127.0.0.1:5000/
    def root():
        # we must create the database
        DB.drop_all()
        DB.create_all()
        # avoiding error since we are dropping all values - no duplicate users
        insert_example_users()
        # renders base.html template and passes down title and users
        return render_template('base.html', title="home", users=User.query.all())

    return app