import docker 
import tempfile #temp directory module
import os 
import shutil



# ignore irrelevant files and folders
# i think this should be dynamic according to the project type, but for now we can just hardcode some common ones
IGNORE_LIST = ['.git', 'node_modules', '__pycache__']

# create a temporary directory for the sandbox
def create_sandbox():
    sandbox_dir = tempfile.TemporaryDirectory()
    return sandbox_dir

# copy the project files to the sandbox, ignoring irrelevant files and folders
def copy_project_to_sandbox(project_path, sandbox_dir):
    for item in os.listdir(project_path):
        if item not in IGNORE_LIST:
            source = os.path.join(project_path, item)
            destination = os.path.join(sandbox_dir, item)
            if os.path.isdir(source):
                # but we need to ignore the ignored folders and files in the subdirectories as well, so we need to use shutil.copytree with ignore parameter
                
                shutil.copytree(source, destination, ignore=shutil.ignore_patterns(*IGNORE_LIST),copy_function=shutil.copy2)
            else:
                # copy2 basically replaces the destination file if it already exists, and also preserves the file metadata, which is useful for our case to avoid any issues with file permissions or timestamps in the sandbox
                shutil.copy2(source, destination)

def inject_ai_patch(sandbox_dir, file_to_fix, patched_code_string):
    """
    Overwrites the broken file in the sandbox with the AI's fix.
    """
    full_path = os.path.join(sandbox_dir, file_to_fix)
    # Ensure the directory exists if the AI is fixing a nesteda file
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    
    with open(full_path, "w") as f:
        f.write(patched_code_string)

# run the tests in the sandbox using Docker
def run_tests_in_sandbox(sandbox_dir, test_command):
    client = docker.from_env()
    
    container = client.containers.run(
        "python:3.11-slim",
        command=test_command,
        volumes={sandbox_dir: {'bind': '/app', 'mode': 'rw'}},
        working_dir='/app',
        detach=True,
        network_disabled=True, # disable network for security reasons
    )
    
    # here first let the container run and also get some logs 
    # if the code fails in the container, we want to see the logs to understand what went wrong
    logs = container.logs(stream=True)

    # we can print logs in real time and we must also save it some where to analyze it later if needed
    for log in logs:
        print(log.decode('utf-8')) # print logs in real time
        with open(os.path.join(sandbox_dir, 'test_logs.txt'), 'a') as log_file:
            log_file.write(log.decode('utf-8')) # save logs to a file
    
    # Grab the logs as a single string so the AI can read it
    output_logs = container.logs(stream=False).decode('utf-8')
    
    container.wait() # wait for the container to finish
    container.reload() # reload the container to get the updated status and exit code
    
    exit_code = container.attrs['State']['ExitCode']
    
    # Aggressively delete the container so we don't leak memory
    container.remove(force=True)
    
    # Return a clean dictionary to the LangGraph Agent!
    return {
        "success": exit_code == 0,
        "logs": output_logs
    }
        
    
    
# clean up the sandbox after running the tests
def clean_up_sandbox(sandbox_dir):
    shutil.rmtree(sandbox_dir)
    
# main function to run the tests in the sandbox
def run_tests(project_path, test_command, test_code_files, patched_code_string, file_to_fix):
    
    # The 'with' block acts as both your creator and your cleanup!
    # 'sandbox_dir' is now safely a string path we can use.
    with tempfile.TemporaryDirectory() as sandbox_dir:
        print(f"Spinning up secure sandbox at {sandbox_dir}...")
        
        # 1. Copy the broken project
        copy_project_to_sandbox(project_path, sandbox_dir)
        
        # 2. Inject the AI's fix
        inject_ai_patch(sandbox_dir, file_to_fix, patched_code_string) 
        
        # 3. Run the tests and grab the dictionary
        result = run_tests_in_sandbox(sandbox_dir, test_command)
        
        # 4. Return the result up to the AI
        return result

# example usage
if __name__ == "__main__":
    project_path = "/path/to/your/project"
    test_command = "pytest" # or any other test command you want to run
    test_code_files = ["test_example.py"] # list of test files to inject
    patched_code_string = "def test_example():\n    assert True" # the patched code string
    file_to_fix = "example.py" # the file to fix
    run_tests(project_path, test_command, test_code_files, patched_code_string, file_to_fix)
    
    
