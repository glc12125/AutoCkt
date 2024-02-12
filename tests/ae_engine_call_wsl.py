from subprocess import check_output

ae_engine_output = check_output(r'"/mnt/c/siemens/SystemExplorer/Automation_Engine/AE_Engine.exe" all --working_dir "/mnt/c/Users/Administrator/workspace/simple_example_modified/ae_run/SimpleExample/WD" --root_dir "/mnt/c/siemens/SystemExplorer" --xml_file "/mnt/c/Users/Administrator/workspace/simple_example_modified/SimpleExample.xml" --xsd_schema "/mnt/c/siemens/SystemExplorer/config/VSE_XSD_Schema/S2S_VSE_XSD_schema.xsd" --vista "/mnt/c/siemens/VirtualPlatform" --nucleus "/mnt/c/siemens"')
print(ae_engine_output)