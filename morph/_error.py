class ValidationError(Exception):
    """Custom error that represents a validation issue, according to internal
    system rules
    """
    def __init__(self, msg):
        super(ValidationError, self).__init__(msg)
