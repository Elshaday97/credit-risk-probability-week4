def handle_errors(func):
    """
    Docstring for handle_errors

    :param func: Function to be decorated
    :return: Wrapped function with error handling
    """

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"[ERROR] {func.__name__}: {e}")
            raise e

    return wrapper
