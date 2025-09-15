import pandas as pd
import xml.etree.ElementTree as ET
import sys

xml_file = sys.argv[1]
tree = ET.parse(xml_file)
root = tree.getroot()

records = []
for record in root.findall('record'):
    record_dict = {}
    for child in record:
        record_dict[child.tag] = child.text
    if record_dict:
        records.append(record_dict)

df = pd.DataFrame(records)
output_file = xml_file.replace('.xml', '.xlsx')
df.to_excel(output_file, index=False)
print(f"转换完成: {output_file}")
