import xmlschema
from pprint import pprint
from xmlschema import XMLSchema11


class AE_Designer:
    def __init__(self, schema_path="./S2S_VSE_XSD_schema.xsd", design_file_path="design.xml"):
        self.schema_path = schema_path
        self.design_file_path = design_file_path
        self.validate()

    def validate(self):
        print("Validating schema.")
        xsd = xmlschema.XMLSchema11(self.schema_path)
        obj = XMLSchema11.meta_schema.decode(self.schema_path)
        #pprint(obj)
        if xsd.is_valid(self.design_file_path):
            print("{} is valided".format(self.design_file_path))
        else:
            print("invalid xml")
        xsd.validate(self.design_file_path)


def main():
    designer = AE_Designer(design_file_path="data/simple_10000_load_sim_dir/SimpleExample-10000.xml")
    #designer = AE_Designer(schema_path="collection_test.xsd", design_file_path="collection_test.xml")

if __name__ == "__main__":
    main()