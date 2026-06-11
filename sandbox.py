import docker
import tempfile
import os
import shutil
import time

def log(level: str, msg: str, *args) -> None:
    prefix = time.strftime("%Y-%m-%d %H:%M:%S")
    try:
        print(f"{prefix} {level}: {msg % args}")
    except Exception:
        print(f"{prefix} {level}: {msg}")

IGNORE_LIST = ['.git', 'node_modules', '__pycache__']

def create_sandbox():
    sandbox_dir = tempfile.TemporaryDirectory()
    return sandbox_dir

def copy_project_to_sandbox(project_path, sandbox_dir):
    for item in os.listdir(project_path):
        if item not in IGNORE_LIST:
            source = os.path.join(project_path, item)
            destination = os.path.join(sandbox_dir, item)
            if os.path.isdir(source):
                shutil.copytree(source, destination, ignore=shutil.ignore_patterns(*IGNORE_LIST), copy_function=shutil.copy2)
            else:
                shutil.copy2(source, destination)

def inject_ai_patch(sandbox_dir, file_to_fix, patched_code_string):
    """
    Overwrites the broken file in the sandbox with the AI's fix.
    """
    full_path = os.path.join(sandbox_dir, file_to_fix)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    with open(full_path, "w") as f:
        f.write(patched_code_string)

def inject_ai_patches(sandbox_dir, patches_dict):
    """
    Inject ALL accumulated patches into the sandbox.
    patches_dict: {"filepath": "patched_code", ...}
    This ensures every fix the agent has ever made is present
    when running tests, preventing the 'amnesia' problem.
    """
    for file_path, code in patches_dict.items():
        full_path = os.path.join(sandbox_dir, file_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "w") as f:
            f.write(code)
        log('INFO', "Injected patch for %s into sandbox", file_path)

def run_tests_in_sandbox(sandbox_dir, test_command):
    client = docker.from_env()
    base_image = "python:3.11-slim"
    tmp_image_tag = "agentic-sandbox:latest"

    install_container = None
    test_container = None

    try:
        req_file = os.path.join(sandbox_dir, "requirements.txt")
        if os.path.exists(req_file):
            log('INFO', "Phase 1: Installing dependencies (network enabled)...")
            install_cmd = "pip install --no-cache-dir -r /app/requirements.txt"
            install_container = client.containers.run(
                base_image,
                command=f"sh -c '{install_cmd}'",
                volumes={sandbox_dir: {'bind': '/app', 'mode': 'ro'}},
                working_dir='/app',
                detach=True,
                network_disabled=False,
            )
            try:
                install_container.wait(timeout=300)
            except Exception as e:
                log('ERROR', "Phase 1 timed out or failed: %s", str(e))
                return {"success": False, "logs": f"Dependency install timed out: {str(e)}"}

            install_container.reload()
            install_exit = install_container.attrs['State']['ExitCode']
            install_logs = install_container.logs(stream=False).decode('utf-8')

            if install_exit != 0:
                return {"success": False, "logs": f"Dependency install failed:\n{install_logs}"}

            install_container.commit(repository="agentic-sandbox", tag="latest")
            log('INFO', "Phase 1 complete: dependencies installed.")
        else:
            tmp_image_tag = base_image

        log('INFO', "Phase 2: Running tests (network disabled)...")
        test_container = client.containers.run(
            tmp_image_tag,
            command=test_command,
            volumes={sandbox_dir: {'bind': '/app', 'mode': 'rw'}},
            working_dir='/app',
            detach=True,
            network_disabled=True,
        )

        output_logs_list = []
        for line in test_container.logs(stream=True):
            decoded = line.decode('utf-8')
            print(decoded, end='')
            output_logs_list.append(decoded)
            with open(os.path.join(sandbox_dir, 'test_logs.txt'), 'a') as log_file:
                log_file.write(decoded)

        test_container.wait(timeout=300)
        test_container.reload()
        exit_code = test_container.attrs['State']['ExitCode']
        output_logs = "".join(output_logs_list)

        try:
            client.containers.run(
                tmp_image_tag,
                command="chmod -R 777 /app",
                volumes={sandbox_dir: {'bind': '/app', 'mode': 'rw'}},
                working_dir='/app',
                detach=False,
                remove=True,
                network_disabled=True,
            )
        except Exception as e:
            log('WARNING', "chmod cleanup step failed (non-fatal): %s", str(e))

        return {"success": exit_code == 0, "logs": output_logs}

    finally:
        for c in [install_container, test_container]:
            if c:
                try:
                    c.remove(force=True)
                except Exception:
                    pass
        try:
            client.images.remove(tmp_image_tag, force=True)
        except Exception:
            pass

def clean_up_sandbox(sandbox_dir):
    shutil.rmtree(sandbox_dir)

def run_tests(project_path, test_command, test_code_files=None,
              patched_code_string=None, file_to_fix=None, patches_dict=None):
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as sandbox_dir:
        print(f"Spinning up secure sandbox at {sandbox_dir}...")
        copy_project_to_sandbox(project_path, sandbox_dir)

        if patches_dict:
            inject_ai_patches(sandbox_dir, patches_dict)
        elif patched_code_string and file_to_fix:
            inject_ai_patch(sandbox_dir, file_to_fix, patched_code_string)

        result = run_tests_in_sandbox(sandbox_dir, test_command)
        return result

if __name__ == "__main__":
    project_path = "/path/to/your/project"
    test_command = "pytest"
    test_code_files = ["tests/test_todos.py"]
    patched_code_string = "def test_create_todo_success():\n    assert True"
    file_to_fix = "main.py"
    run_tests(project_path, test_command, test_code_files, patched_code_string, file_to_fix)
