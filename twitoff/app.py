"""Main app/routing file for Twitoff"""

from os import getenv
from flask import Flask, render_template
from .models import DB, User
from .twitter import add_or_update_user


# creates application
def create_app():
    """Creating and configuring an instance of the Flask application"""
    app = Flask(__name__)

    # database and app configurations
    app.config["SQLALCHEMY_DATABASE_URI"] = getenv("DATABASE_URI")
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

    # initilizing database
    DB.init_app(app)

    # decorator listens for specific endpoint visits
    @app.route('/')  # http://127.0.0.1:5000/
    def root():
        # renders base.html template and passes down title and users
        return render_template('base.html', title="Home", users=User.query.all())

    @app.route('/update')  # http://127.0.0.1:5000/update
    def update():
        insert_example_users()
        return render_template('base.html', title="Home", users=User.query.all())

    @app.route('/reset')  # http://127.0.0.1:5000/reset
    def reset():
        # we must create the database
        DB.drop_all()
        DB.create_all()
        return render_template('base.html', title="Home")
    return app


def insert_example_users():
    add_or_update_user('elonmusk')
    add_or_update_user('nasa')