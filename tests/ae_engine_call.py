from subprocess import check_output

ae_engine_output = check_output(r'"C:\siemens\SystemExplorer\Automation_Engine\AE_Engine.exe" all --working_dir "C:\Users\Administrator\workspace\simple_example_modified\ae_run\SimpleExample\WD" --root_dir "C:\siemens\SystemExplorer" --xml_file "C:/Users/Administrator/workspace/simple_example_modified/ae_run/SimpleExample.xml" --xsd_schema "C:\siemens\SystemExplorer\config/VSE_XSD_Schema/S2S_VSE_XSD_schema.xsd" --vista "C:\siemens\VirtualPlatform" --nucleus "C:\siemens"')
print(ae_engine_output)