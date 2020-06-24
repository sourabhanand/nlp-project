from api.app import create_app


application = create_app()
application.config['JSONIFY_PRETTYPRINT_REGULAR'] = True