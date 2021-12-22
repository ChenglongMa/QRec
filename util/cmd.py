import sys


def get_argv(idx=1, defaults=None):
    # argv index starts from 1
    idx = 1 if idx < 1 else idx
    res = sys.argv[idx] if len(sys.argv) > idx else defaults
    if isinstance(defaults, bool):
        return str(res).lower() in ['true', '1', 't', 'y', 'yes', 'yeah', 'yup', 'certainly']
    if isinstance(defaults, int):
        return int(res)
    else:
        return res
