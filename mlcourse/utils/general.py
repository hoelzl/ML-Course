
from pprint import pprint
from io import StringIO

# %%
def pprint_indent(x, indent=0):
    stream = StringIO()
    pprint(x, stream=stream, width=120 - indent)
    lines = stream.getvalue().split('\n')
    if not lines:
        return
    first_line = lines[0]
    rest_lines = (' ' * indent + line for line in lines[1:])
    print(first_line)
    for line in rest_lines:
        print(line)
