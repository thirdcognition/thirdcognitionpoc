import pprint as pp
from lib.load_env import DEBUGMODE

def print_params(msg = "", params = ""):
    if DEBUGMODE:
        if msg: print(f"\n\n\n{msg}")
        if params: print(f"'\n\n{pp.pformat(params).replace("\\n", "\n")}\n\n")