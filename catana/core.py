
def use_signature(Obj):
    """Apply call signature and documentation of Obj to the decorated method"""
    def decorate(f):
        # call-signature of f is exposed via __wrapped__.
        # we want it to mimic Obj.__init__
        f.__wrapped__ = Obj.__init__
        f._uses_signature = Obj

        # Supplement the docstring of f with information from Obj
        if Obj.__doc__:
            doclines = Obj.__doc__.splitlines()
            if f.__doc__:
                doc = f.__doc__ + '\n'.join(doclines[1:])
            else:
                doc = '\n'.join(doclines)
            try:
                f.__doc__ = doc
            except AttributeError:
                # __doc__ is not modifiable for classes in Python < 3.3
                pass

        return f
    return decorate
