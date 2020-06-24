from flask import Flask

from controller import extraction_app


def create_app() -> Flask:
    """Creating a flask app instance."""

    flask_app = Flask('extraction_api', template_folder='./template')
    flask_app.secret_key = "super secret key"
    # import blueprints
    flask_app.register_blueprint(extraction_app)

    return flask_app

