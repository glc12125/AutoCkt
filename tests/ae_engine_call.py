import os
from subprocess import check_output
import shutil
from pathlib import Path
import time

os.environ["SALT_LICENSE_SERVER"] = "C:\\development\\AE\\Architecture_Explorer\\license_DESKTOP-P509B0S_standalone.lic"


#shutil.rmtree('C:\\development\\AE\\Architecture_Explorer\\AutoCkt\\ae_data\\designs_SimpleExample\\SimpleExample_1_4_200000000_1_5364\\ae_run\\WD')
#os.system('rmdir /S /Q "{}"'.format('C:\\development\\AE\\Architecture_Explorer\\AutoCkt\\ae_data\\designs_SimpleExample\\SimpleExample_1_4_200000000_1_5364\\ae_run\\WD'))
#Path('C:\\development\\AE\\Architecture_Explorer\\AutoCkt\\ae_data\\designs_SimpleExample\\SimpleExample_1_4_200000000_1_5364\\ae_run\\WD').mkdir(parents=True, exist_ok=True)
design_folder = "C:\\development\\AE\\Architecture_Explorer\\AutoCkt\\ae_data\\designs_SimpleExample\\SimpleExample_1_4_200000000_1_5364"
workding_directory_path = os.path.join(design_folder, "ae_run", "WD")
Path(workding_directory_path).mkdir(parents=True, exist_ok=True)

print("Starting AE Engine")
start_time = time.time()
#workding_directory = "C:\\development\\AE\\Architecture_Explorer\\AutoCkt\\ae_data\\designs_SimpleExample\\SimpleExample_1_4_200000000_1_5364\\ae_run\\WD"
design_file = "C:\\development\\AE\\Architecture_Explorer\\AutoCkt\\ae_data\\designs_SimpleExample\\SimpleExample_1_4_200000000_1_5364\\SimpleExample_1_4_200000000_1.xml"
ae_engine_output = check_output("C:\\siemens\\SystemExplorer\\Automation_Engine\\AE_Engine.exe all --working_dir %s --root_dir C:\\siemens\\SystemExplorer --xml_file %s --xsd_schema C:\\siemens\\SystemExplorer\\config\\VSE_XSD_Schema\\S2S_VSE_XSD_schema.xsd --vista C:\\siemens\\VirtualPlatform --nucleus C:\\siemens" % (workding_directory_path, design_file))
end_time = time.time()
print("Done")
print("AE_Engine took {} seconds".format(end_time - start_time))
print(ae_engine_output)