import sys
from PIL import Image

error_file_path = sys.argv[1]
with open(error_file_path) as f:
    errors = f.read()
errors = errors.split("\n")

for i in range(len(errors)):
    if i % 3 != 0:
        continue
    filename = errors[i]
    actual = errors[i + 1]
    im = Image.open(filename)
    im.show()
    print(actual)
    raw_input()
    im.close()
