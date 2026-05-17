from __future__ import annotations
import os
import json
import time
import re
import ast
from pathlib import Path
from typing import List, Dict, Any, TypedDict, Optional
from langgraph.graph import StateGraph, START, END
from github import Github

# 1. IMPORT THE LANGCHAIN UNIFIED INTERFACE
from langchain_google_genai import ChatGoogleGenerativeAI

# Import your sandbox
from sandbox import run_tests

def log(level: str, msg: str, *args) -> None:
    prefix = time.strftime("%Y-%m-%d %H:%M:%S")
    try:
        print(f"{prefix} {level}: {msg % args}")
    except Exception:
        print(f"{prefix} {level}: {msg}")

MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "3"))
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.5-flash") # Using flash or pro
TEMPERATURE = float(os.getenv("GEMINI_TEMP", "0.2"))
ALLOWED_ACTIONS = set([a.strip() for a in os.getenv("ALLOWED_ACTIONS", "create_pr").split(",") if a.strip()])

llm = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=TEMPERATURE)

AGENTIC_TMP_DIR = Path('agentic_tmp')
AGENTIC_TMP_DIR.mkdir(exist_ok=True)

FENCE_RE = re.compile(r"```(?:json|python)?\s*(.*?)\s*```", re.S)

def extract_text(response: Any) -> str:
    content = getattr(response, 'content', str(response))
    if isinstance(content, list):
        return "".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in content
        ).strip()
    return str(content).strip()

def strip_fences(text: str) -> str:
    if not text:
        return text
    m = FENCE_RE.search(text)
    return m.group(1).strip() if m else text.strip()

def safe_load_json(text: str) -> Any:
    text = strip_fences(text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start:end+1])
            except json.JSONDecodeError:
                pass
    raise ValueError("Could not parse JSON from LLM output")

def invoke_with_retries(prompt: str, retries: int = 3, delay: float = 1.0) -> Any:
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            return llm.invoke(prompt)
        except Exception as e:
            last_exc = e
            time.sleep(delay * attempt)
    raise last_exc

def save_audit(iteration: int, name: str, prompt: str, response_text: str) -> Path:
    path = AGENTIC_TMP_DIR / f"iter_{iteration}_{name}.json"
    data = {
        'iteration': iteration,
        'name': name,
        'prompt': prompt,
        'response': response_text,
        'timestamp': time.time()
    }
    path.write_text(json.dumps(data, indent=2))
    return path

# ------------------
# Helpers for v2
# ------------------

def apply_edits(original_code: str, edits: List[dict]) -> str:
    lines = original_code.split('\n')
    # Sort edits in reverse order so line numbers don't shift for earlier edits
    edits = sorted(edits, key=lambda x: x.get('start_line', 0), reverse=True)
    for edit in edits:
        start = edit['start_line'] - 1
        end = edit['end_line']
        replacement = edit['replacement'].split('\n')
        lines[start:end] = replacement
    return '\n'.join(lines)

def find_callers(target_file: str) -> dict:
    target_module = Path(target_file).stem
    callers = {}
    for py_file in Path('.').rglob('*.py'):
        if any(part.startswith('.') or part in ('__pycache__', '.venv', 'node_modules') for part in py_file.parts):
            continue
        if str(py_file) == target_file:
            continue
        try:
            source = py_file.read_text()
            if f"from {target_module} import" in source or f"import {target_module}" in source:
                callers[str(py_file)] = source
        except Exception:
            pass
    return callers

# ------------------
# GitHub Helpers
# ------------------
def github_client():
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        raise RuntimeError("GITHUB_TOKEN is required for GitHub operations")
    from github import Auth
    return Github(auth=Auth.Token(token))

def create_branch_and_commit_multiple(repo_full_name: str, branch_name: str, patches_dict: dict, commit_message: str):
    gh = github_client()
    repo = gh.get_repo(repo_full_name)
    base_branch = os.getenv("GITHUB_BASE_BRANCH", "main")
    base_ref = repo.get_git_ref(f"heads/{base_branch}")
    try:
        existing_ref = repo.get_git_ref(f"heads/{branch_name}")
        existing_ref.delete()
        log('INFO', "Deleted stale branch %s.", branch_name)
    except Exception:
        pass
    repo.create_git_ref(ref=f"refs/heads/{branch_name}", sha=base_ref.object.sha)
    log('INFO', "Created branch %s from %s.", branch_name, base_branch)
    
    for file_path, file_content in patches_dict.items():
        existing_sha = None
        try:
            existing = repo.get_contents(file_path, ref=branch_name)
            existing_sha = existing.sha
            log('INFO', "Updating %s on branch %s.", file_path, branch_name)
            repo.update_file(path=file_path, message=commit_message, content=file_content, sha=existing_sha, branch=branch_name)
        except Exception:
            log('INFO', "Creating %s on branch %s.", file_path, branch_name)
            repo.create_file(path=file_path, message=commit_message, content=file_content, branch=branch_name)

def open_pr_with_rca(repo_full_name: str, branch_name: str, pr_title: str, pr_body: str) -> str:
    gh = github_client()
    repo = gh.get_repo(repo_full_name)
    base_branch = os.getenv("GITHUB_BASE_BRANCH", "main")
    pr = repo.create_pull(title=pr_title, body=pr_body, head=branch_name, base=base_branch)
    return pr.html_url

# ==========================================
# THE STATE 
# ==========================================
class AgenticState(TypedDict):
    logs: str                  
    iteration_count: int       
    success: bool                
    
    # Memory Bank
    repair_memory: Dict[str, Any]
    
    current_file: Optional[str]
    repair_strategy: Optional[str]
    
    rca_html_path: Optional[str]
    pr_url: Optional[str]
    approved: Optional[bool]
    lint_failed: Optional[bool]

# ==========================================
# THE NODES
# ==========================================
def fetch_logs_node(state: AgenticState) -> dict:    
    log('INFO', "NODE[fetch_logs_node]: Fetching logs...")
    
    # Initialize repair memory if it doesn't exist
    repair_memory = state.get("repair_memory")
    if not repair_memory:
        repair_memory = {
            "iterations": [],
            "repo_state": {},
            "context": {
                "original_logs": "",
                "latest_logs": "",
                "files_attempted": []
            }
        }

    if state.get("iteration_count", 0) > 0:
        log('INFO', "-> Fetching latest test logs from the secure Sandbox.")
        repair_memory["context"]["latest_logs"] = state["logs"]
        return {"logs": state["logs"], "repair_memory": repair_memory}

    # Fetch real logs from GitHub Actions
    workflow_run_id = os.getenv("workflow_run_id")
    repo_full_name = os.getenv("GITHUB_REPO") or os.getenv("GITHUB_REPOSITORY")

    if repo_full_name:
        try:
            gh = github_client()
            repo = gh.get_repo(repo_full_name)

            run = None
            if workflow_run_id:
                run = repo.get_workflow_run(int(workflow_run_id))
            else:
                failed_runs = repo.get_workflow_runs(status="failure")
                for candidate in failed_runs:
                    if candidate.name and "doctor" in candidate.name.lower():
                        continue
                    run = candidate
                    break

            if run is None:
                return {"logs": "No failed workflow runs found.", "repair_memory": repair_memory}

            logs_parts = []
            for job in run.jobs():
                if job.conclusion == "failure":
                    for step in job.steps:
                        if step.conclusion == "failure":
                            logs_parts.append(f"=== Job: {job.name} | Step: {step.name} ===\nStatus: {step.conclusion}\n")
            
            import requests as req
            import zipfile
            import io
            token = os.getenv("GITHUB_TOKEN")
            headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3+json"}
            logs_url = f"https://api.github.com/repos/{repo_full_name}/actions/runs/{run.id}/logs"
            resp = req.get(logs_url, headers=headers, allow_redirects=True)
            if resp.status_code == 200:
                z = zipfile.ZipFile(io.BytesIO(resp.content))
                for name in z.namelist():
                    content = z.read(name).decode("utf-8", errors="replace")
                    if any(kw in name.lower() for kw in ["run tests", "test", "build", "run"]):
                        logs_parts.append(f"=== {name} ===\n{content[-3000:]}\n")
                    elif not logs_parts:
                        logs_parts.append(f"=== {name} ===\n{content[-2000:]}\n")
            
            if logs_parts:
                real_logs = "\n".join(logs_parts)
                if len(real_logs) > 8000:
                    real_logs = real_logs[-8000:]
                repair_memory["context"]["original_logs"] = real_logs
                repair_memory["context"]["latest_logs"] = real_logs
                return {"logs": real_logs, "repair_memory": repair_memory}
            else:
                return {"logs": "Workflow run found but no failure logs could be extracted.", "repair_memory": repair_memory}
        except Exception as e:
            return {"logs": f"Failed to fetch logs from GitHub: {str(e)}", "repair_memory": repair_memory}

    return {"logs": "No GITHUB_REPO configured.", "repair_memory": repair_memory}

def analyze_code_node(state: AgenticState) -> dict:
    log('INFO', "NODE[analyze_code_node]: Planning repair strategy...")
    
    repair_memory = state.get("repair_memory", {})
    iterations_history = json.dumps(repair_memory.get("iterations", []), indent=2)

    repo_files = []
    try:
        for p in Path('.').rglob('*.py'):
            if not any(part.startswith('.') or part in ('__pycache__', '.venv', 'node_modules') for part in p.parts):
                repo_files.append(str(p))
    except Exception:
        pass
    file_listing = ', '.join(repo_files) if repo_files else 'unknown'

    prompt = f"""
    You are a Senior Python Developer diagnosing a CI/CD failure.
    
    ERROR LOGS:
    {state['logs']}

    AVAILABLE FILES IN REPO:
    {file_listing}

    PAST REPAIR ATTEMPTS (Do not repeat failed strategies):
    {iterations_history}

    Analyze the logs and return a JSON object with:
    {{
      "failure_summary": "Short description of the error",
      "root_cause": "Detailed explanation of why it failed",
      "target_file": "The EXACT file path from AVAILABLE FILES to fix",
      "repair_strategy": "Step-by-step plan to fix it",
      "confidence": 0.0 to 1.0
    }}
    """
    
    response = invoke_with_retries(prompt)
    resp_text = extract_text(response)
    save_audit(state.get('iteration_count', 0), 'plan_response', prompt, resp_text)

    try:
        analysis = safe_load_json(resp_text)
        target_file = analysis.get("target_file", "")
        repair_strategy = analysis.get("repair_strategy", "")
        
        # Track attempted files
        if target_file and target_file not in repair_memory["context"]["files_attempted"]:
            repair_memory["context"]["files_attempted"].append(target_file)
            
        log('INFO', "-> Plan: Fix %s | Strategy: %s", target_file, repair_strategy)
        return {
            "current_file": target_file,
            "repair_strategy": repair_strategy,
            "repair_memory": repair_memory
        }
    except Exception as e:
        log('ERROR', "Failed to parse analysis JSON: %s", str(e))
        return {"current_file": "", "repair_strategy": "Failed to parse plan."}

def fix_code_node(state: AgenticState) -> dict:
    current_file = state.get("current_file")
    iteration = state.get("iteration_count", 0)
    repair_memory = state.get("repair_memory", {})
    repo_state = repair_memory.get("repo_state", {})
    
    log('INFO', "NODE[fix_code_node]: Generating surgical patch for %s", current_file)

    if not current_file:
        return {"lint_failed": False}

    # 1. Read from repo_state memory OR disk
    if current_file in repo_state:
        broken_code = repo_state[current_file]
        log('INFO', "-> Reading previously patched version of %s from memory", current_file)
    else:
        try:
            broken_code = Path(current_file).read_text()
        except Exception:
            broken_code = "# File missing or empty"

    # 2. Context-aware reading (callers)
    callers = find_callers(current_file)
    callers_context = ""
    if callers:
        callers_context = "CALLERS OF THIS FILE (for context):\n"
        for f, code in callers.items():
            callers_context += f"--- {f} ---\n{code[-1000:]}\n\n"

    # 3. Prompt for surgical diff
    prompt = f"""
    You are a Senior Python Developer implementing a fix.

    TARGET FILE: {current_file}
    REPAIR STRATEGY: {state.get('repair_strategy')}
    ERROR LOGS: {state.get('logs')}

    {callers_context}

    CURRENT CODE ({current_file}):
    {broken_code}

    Return ONLY a JSON object with your surgical edits. Do NOT rewrite the whole file unless necessary.
    {{
      "edits": [
        {{
          "start_line": <int> (1-indexed),
          "end_line": <int> (1-indexed, inclusive),
          "replacement": "<new code for these lines>"
        }}
      ]
    }}
    """

    response = invoke_with_retries(prompt)
    resp_text = extract_text(response)
    save_audit(iteration, 'fix_response', prompt, resp_text)

    try:
        edits_json = safe_load_json(resp_text)
        edits = edits_json.get("edits", [])
        
        # Apply the edits programmatically
        patched_code = apply_edits(broken_code, edits)
        repo_state[current_file] = patched_code
        
        # Log to repair memory
        repair_memory["iterations"].append({
            "iteration": iteration + 1,
            "target_file": current_file,
            "strategy": state.get("repair_strategy"),
            "edits_applied": edits,
            "result": "pending",
            "reason": None,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
        })
        
        log('INFO', "-> Applied %d edits to %s", len(edits), current_file)
        
    except Exception as e:
        log('ERROR', "Failed to apply surgical edits, falling back to original code: %s", str(e))
        # If it hallucinates non-JSON, we might try to save it directly if it looks like code
        raw = strip_fences(resp_text)
        if "def " in raw or "import " in raw or "class " in raw:
             repo_state[current_file] = raw
             log('INFO', "-> Fallback: Applied raw code block to %s", current_file)
             repair_memory["iterations"].append({
                 "iteration": iteration + 1,
                 "target_file": current_file,
                 "strategy": state.get("repair_strategy"),
                 "edits_applied": "Raw file rewrite fallback",
                 "result": "pending",
                 "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
             })

    repair_memory["repo_state"] = repo_state
    
    return {
        "repair_memory": repair_memory,
        "iteration_count": iteration + 1,
        "lint_failed": False # Reset lint status
    }

def lint_check_node(state: AgenticState) -> dict:
    log('INFO', "NODE[lint_check_node]: Running syntax checks on patched files...")
    import subprocess
    repo_state = state.get("repair_memory", {}).get("repo_state", {})
    
    for filepath, code in repo_state.items():
        tmp = Path(f"/tmp/lint_agentic.py")
        tmp.write_text(code)
        result = subprocess.run(["python", "-m", "py_compile", str(tmp)], capture_output=True, text=True)
        if result.returncode != 0:
            log('WARNING', "Syntax error found in %s:\n%s", filepath, result.stderr)
            # Record failure in memory
            iterations = state.get("repair_memory", {}).get("iterations", [])
            if iterations:
                iterations[-1]["result"] = "failed"
                iterations[-1]["reason"] = f"Syntax error: {result.stderr}"
            return {"lint_failed": True, "logs": f"Syntax error in {filepath}:\n{result.stderr}"}
            
    log('INFO', "-> All patched files passed syntax checks.")
    return {"lint_failed": False}

def test_code_node(state: AgenticState) -> dict:
    log('INFO', "NODE[test_code_node]: Injecting ALL accumulated patches into Sandbox...")
    
    repair_memory = state.get("repair_memory", {})
    repo_state = repair_memory.get("repo_state", {})
    
    sandbox_result = run_tests(
        project_path=".",
        test_command="python -m pytest tests/ -v",
        patches_dict=repo_state
    )

    success = sandbox_result.get("success", False)
    logs = sandbox_result.get("logs", "")
    
    if success:
        log('INFO', "-> Sandbox Execution: SUCCESS")
    else:
        log('WARNING', "-> Sandbox Execution: FAILED. Gathering new logs...")

    # Update iteration history
    iterations = repair_memory.get("iterations", [])
    if iterations:
        iterations[-1]["result"] = "passed" if success else "failed"
        iterations[-1]["reason"] = "Tests failed" if not success else "Tests passed"

    return {
        "success": success,
        "logs": logs,
        "repair_memory": repair_memory
    }

def generate_rca_node(state: AgenticState) -> dict:
    log('INFO', "NODE[generate_rca_node]: Generating RCA and HTML Report...")
    repair_memory = state.get("repair_memory", {})
    iterations = repair_memory.get("iterations", [])
    
    # 1. JSON RCA for PR body
    rca_obj = {
        "summary": "Automated pipeline repair",
        "iterations_taken": len(iterations),
        "files_modified": list(repair_memory.get("repo_state", {}).keys()),
        "final_status": "Success" if state.get("success") else "Max Iterations Reached"
    }
    rca_path = AGENTIC_TMP_DIR / "final_rca.json"
    rca_path.write_text(json.dumps(rca_obj, indent=2))
    
    # 2. HTML Report
    html_content = f"""
    <html>
    <head><title>Pipeline Doctor Report</title><style>body{{font-family:sans-serif;}} .iter{{border:1px solid #ccc; margin:10px; padding:10px;}}</style></head>
    <body>
    <h1>Pipeline Doctor - Repair Timeline</h1>
    <h3>Status: {rca_obj['final_status']}</h3>
    <p>Files Patched: {', '.join(rca_obj['files_modified'])}</p>
    """
    for it in iterations:
        html_content += f"""
        <div class='iter'>
            <h4>Iteration {it.get('iteration')} - {it.get('target_file')}</h4>
            <p><b>Strategy:</b> {it.get('strategy')}</p>
            <p><b>Result:</b> {it.get('result')} ({it.get('reason')})</p>
        </div>
        """
    html_content += "</body></html>"
    
    html_path = AGENTIC_TMP_DIR / "report.html"
    html_path.write_text(html_content)
    
    log('INFO', "-> RCA and HTML report generated.")
    return {"rca_path": str(rca_path), "rca_html_path": str(html_path)}

def wait_for_approval_node(state: AgenticState) -> dict:
    hitl = os.getenv("HITL_ENABLED", "true").lower() == "true"
    if not hitl:
        return {"approved": True}
    iteration = state.get("iteration_count", 0)
    marker = AGENTIC_TMP_DIR / f"iter_{iteration}" / "approved"
    timeout = int(os.getenv("HITL_TIMEOUT_SEC", "600"))
    interval = 5
    waited = 0
    while waited < timeout:
        if marker.exists():
            return {"approved": True}
        time.sleep(interval)
        waited += interval
    return {"approved": False}

def create_pr_node(state: AgenticState) -> dict:
    log('INFO', "NODE[create_pr_node]: Creating PR with ALL patches...")
    if 'create_pr' not in ALLOWED_ACTIONS:
        return {"pr_created": False}
        
    repo_state = state.get("repair_memory", {}).get("repo_state", {})
    if not repo_state:
        return {"pr_created": False, "reason": "no-patches"}

    repo_full_name = os.getenv("GITHUB_REPO") or os.getenv("GITHUB_REPOSITORY")
    if not repo_full_name:
        return {"pr_created": False, "reason": "missing-repo"}

    branch_name = f"agentic/auto-fix/run_{int(time.time())}"
    commit_msg = f"agentic: auto-fix ({len(repo_state)} files)"
    
    try:
        create_branch_and_commit_multiple(repo_full_name, branch_name, repo_state, commit_msg)
        
        pr_title = "agentic: Pipeline Auto-Fix"
        pr_body = "## Automated fix from Pipeline Doctor\n\nFixed files:\n"
        for f in repo_state.keys():
            pr_body += f"- `{f}`\n"
        pr_body += "\n_Check the attached HTML report artifact for the full repair timeline._"
        
        pr_url = open_pr_with_rca(repo_full_name, branch_name, pr_title, pr_body)
        log('INFO', "-> PR created: %s", pr_url)
        return {"pr_url": pr_url}
    except Exception as e:
        log('ERROR', "-> Failed to create PR: %s", str(e))
        return {"reason": str(e)}

# ==========================================
# ROUTING & GRAPH
# ==========================================
def route_after_lint(state: AgenticState) -> str:
    if state.get("lint_failed"):
        log('INFO', "[ROUTER]: Lint failed. Looping back to fix_code.")
        return "fix_code"
    return "test_code"

def route_after_test(state: AgenticState) -> str:
    if state.get("success"):
        log('INFO', "[ROUTER]: Tests passed! Moving to RCA/PR.")
        return "generate_rca"
    elif state.get("iteration_count", 0) >= MAX_ITERATIONS:
        log('INFO', "[ROUTER]: Max iterations reached. Generating final RCA.")
        return "generate_rca"
    else:
        log('INFO', "[ROUTER]: Tests failed. Looping back to analyze.")
        return "analyze_code"

def route_after_rca(state: AgenticState) -> str:
    # If hitl is enabled and it was successful
    if state.get("success"):
        return "wait_for_approval"
    return END

def route_after_approval(state: AgenticState) -> str:
    return "create_pr" if state.get("approved", False) else END

agent_builder = StateGraph(AgenticState)

agent_builder.add_node("fetch_logs", fetch_logs_node)
agent_builder.add_node("analyze_code", analyze_code_node) 
agent_builder.add_node("fix_code", fix_code_node)
agent_builder.add_node("lint_check", lint_check_node)
agent_builder.add_node("test_code", test_code_node)
agent_builder.add_node("generate_rca", generate_rca_node)
agent_builder.add_node("wait_for_approval", wait_for_approval_node)
agent_builder.add_node("create_pr", create_pr_node)

agent_builder.add_edge(START, "fetch_logs")
agent_builder.add_edge("fetch_logs", "analyze_code")
agent_builder.add_edge("analyze_code", "fix_code")
agent_builder.add_edge("fix_code", "lint_check")

agent_builder.add_conditional_edges("lint_check", route_after_lint)
agent_builder.add_conditional_edges("test_code", route_after_test)
agent_builder.add_conditional_edges("generate_rca", route_after_rca)
agent_builder.add_conditional_edges("wait_for_approval", route_after_approval)
agent_builder.add_edge("create_pr", END)

agentic_graph = agent_builder.compile()

if __name__ == "__main__":
    log('INFO', "🚀 Starting Pipeline Doctor Agent v2 (powered by LLM)...")
    initial_state = {"iteration_count": 0, "success": False, "logs": ""}
    for step in agentic_graph.stream(initial_state):
        pass