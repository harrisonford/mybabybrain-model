
# A collection of quality of life functions that may be used in many places


# A silly function to set a default value if not in kwargs
def kwarget(key, default, **kwargs):
    if key in kwargs:
        return kwargs[key]
    else:
        return default
