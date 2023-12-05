class CoalitionNotFoundException(Exception):
    """Raised when coalition does not exist"""
    def __init__(self, message="Coalition does not exist!"):
        self.message = message
        super().__init__(self.message)
