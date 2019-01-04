

class Printer(object):

    def __init__(self, output_filename, fields):
        self.file = open(output_filename, "w+")
        self.fields = fields

    def write_header(self):
        self.file.write("field_unique_id\t")
        for field in self.fields:
            self.file.write(field + "\t")
        self.file.write('\n')

    def write_line(self, unique_id, line):
        self.file.write(unique_id + '\t')
        for i, field in enumerate(self.fields):
            self.file.write(line[field])
            if i != len(self.fields) - 1:
                self.file.write('\t')
        self.file.write('\n')

    def close(self):
        self.file.close()
