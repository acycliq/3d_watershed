import logging

# Create our demo logger
logger = logging.getLogger('3d_watershed')
# Set a log level for the logger
logger.setLevel(logging.INFO)
# Create a console handler
handler = logging.StreamHandler()
# Set INFO level for handler
handler.setLevel(logging.INFO)
# Create a message format that matches earlier example
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# Add our format to our handler
handler.setFormatter(formatter)
# Add our handler to our logger
logger.addHandler(handler)
