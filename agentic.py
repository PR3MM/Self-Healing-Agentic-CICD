from __future__ import annotations
import os
import json
import time
import re
from pathlib import Path
from typing import List, Dict, Any, TypedDict, Optional
from langgraph.graph import StateGraph, START, END
from github import Github

# 1. IMPORT THE LANGCHAIN UNIFIED INTERFACE
from langchain_google_genai import ChatGoogleGenerativeAI

# Import your sandbox
from sandbox import run_tests

# Simple print-based logger (avoid using `logging` module per user preference)
def log(level: str, msg: str, *args) -> None:
    prefix = time.strftime("%Y-%m-%d %H:%M:%S")
    try:
        print(f"{prefix} {level}: {msg % args}")
    except Exception:
        print(f"{prefix} {level}: {msg}")

# Load configuration from env with sane defaults
MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "3"))
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
TEMPERATURE = float(os.getenv("GEMINI_TEMP", "0.2"))
# Comma-separated allowed actions for guardrails: create_pr, auto_merge, commit
# Default: allow creating PRs (staged patches are still written locally unconditionally)
ALLOWED_ACTIONS = set([a.strip() for a in os.getenv("ALLOWED_ACTIONS", "create_pr").split(",") if a.strip()])

# 2. INITIALIZE LLM
# Make sure export GEMINI_API_KEY="your-key-here" is set in your terminal
llm = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=TEMPERATURE)

# Directory for staging artifacts (patches, audit, RCA)
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
    """Attempt to parse JSON from LLM output, with a small fallback to extract the first {...} substring."""
    text = strip_fences(text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Fallback: find first { ... } block
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start:end+1])
            except json.JSONDecodeError:
                log('DEBUG', "Fallback JSON parse failed")
    raise ValueError("Could not parse JSON from LLM output")

def validate_analysis_schema(obj: Dict[str, Any]) -> bool:
    if not isinstance(obj, dict):
        return False
    return isinstance(obj.get('file_to_fix'), str) and isinstance(obj.get('context'), str)

def validate_rca_schema(obj: Dict[str, Any]) -> bool:
    if not isinstance(obj, dict):
        return False
    return all(isinstance(obj.get(k), str) for k in ('summary','root_cause','patch_summary'))

def invoke_with_retries(prompt: str, retries: int = 3, delay: float = 1.0) -> Any:
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            log('DEBUG', "Invoking LLM (attempt %d)", attempt)
            resp = llm.invoke(prompt)
            return resp
        except Exception as e:
            last_exc = e
            log('WARNING', "LLM invoke failed (attempt %d/%d): %s", attempt, retries, str(e))
            time.sleep(delay * attempt)
    log('ERROR', "LLM invoke exhausted retries")
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

def write_temp_patch(iteration: int, target_file: str, code: str) -> Path:
    safe_dir = AGENTIC_TMP_DIR / f"iter_{iteration}"
    safe_dir.mkdir(parents=True, exist_ok=True)
    p = safe_dir / Path(target_file).name
    p.write_text(code)
    return p


# ------------------
# GitHub + HITL helpers
# ------------------
def github_client():
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        raise RuntimeError("GITHUB_TOKEN is required for GitHub operations")
    from github import Auth
    return Github(auth=Auth.Token(token))

def create_branch_and_commit(repo_full_name: str, branch_name: str, file_path: str, file_content: str, commit_message: str):
    gh = github_client()
    repo = gh.get_repo(repo_full_name)
    base_branch = os.getenv("GITHUB_BASE_BRANCH", "main")
    base_ref = repo.get_git_ref(f"heads/{base_branch}")
    # create branch if missing
    try:
        repo.get_git_ref(f"heads/{branch_name}")
        log('INFO', "Branch %s already exists, reusing it.", branch_name)
    except Exception:
        repo.create_git_ref(ref=f"refs/heads/{branch_name}", sha=base_ref.object.sha)
        log('INFO', "Created new branch %s.", branch_name)
    # Always check the TARGET branch for existing file SHA to avoid 422 errors
    existing_sha = None
    try:
        existing = repo.get_contents(file_path, ref=branch_name)
        existing_sha = existing.sha
        log('INFO', "File %s exists on branch %s, will update it.", file_path, branch_name)
    except Exception:
        log('INFO', "File %s not found on branch %s, will create it.", file_path, branch_name)
    if existing_sha:
        repo.update_file(path=file_path, message=commit_message, content=file_content, sha=existing_sha, branch=branch_name)
    else:
        repo.create_file(path=file_path, message=commit_message, content=file_content, branch=branch_name)

def open_pr_with_rca(repo_full_name: str, branch_name: str, pr_title: str, pr_body: str) -> str:
    gh = github_client()
    repo = gh.get_repo(repo_full_name)
    base_branch = os.getenv("GITHUB_BASE_BRANCH", "main")
    pr = repo.create_pull(title=pr_title, body=pr_body, head=branch_name, base=base_branch)
    return pr.html_url

def wait_for_approval_node(state: AgenticState) -> dict:
    iteration = state.get("iteration_count", 0)
    hitl = os.getenv("HITL_ENABLED", "true").lower() == "true"
    if not hitl:
        return {"approved": True}
    marker = AGENTIC_TMP_DIR / f"iter_{iteration}" / "approved"
    timeout = int(os.getenv("HITL_TIMEOUT_SEC", "600"))
    interval = 5
    waited = 0
    while waited < timeout:
        if marker.exists():
            return {"approved": True}
        time.sleep(interval)
        waited += interval
    return {"approved": False, "reason": "approval_timeout"}

def create_pr_node(state: AgenticState) -> dict:
    log('INFO', "NODE[create_pr_node]: Attempting to create PR...")
    if 'create_pr' not in ALLOWED_ACTIONS:
        log('WARNING', "-> PR skipped: 'create_pr' action not allowed")
        return {"pr_created": False, "reason": "action-not-allowed"}
    iteration = state.get("iteration_count", 0)
    patched_rel = state.get("patched_file_path")
    if not patched_rel:
        log('WARNING', "-> PR skipped: no patched file found in state")
        return {"pr_created": False, "reason": "no-patched-file"}
    # read staged file content from .agentic_tmp
    staged_path = Path(patched_rel)
    if not staged_path.exists():
        log('WARNING', "-> PR skipped: staged file missing at %s", str(staged_path))
        return {"pr_created": False, "reason": "staged-file-missing"}
    content = staged_path.read_text()
    repo_full_name = os.getenv("GITHUB_REPO") or os.getenv("GITHUB_REPOSITORY")
    if not repo_full_name:
        try:
            import subprocess
            origin_url = subprocess.check_output(["git", "config", "--get", "remote.origin.url"]).decode("utf-8").strip()
            if "github.com" in origin_url:
                if origin_url.startswith("https://"):
                    repo_full_name = origin_url.split("github.com/")[-1].replace(".git", "")
                elif origin_url.startswith("git@"):
                    repo_full_name = origin_url.split("github.com:")[-1].replace(".git", "")
        except Exception as e:
            log('DEBUG', "Failed to fetch repo from git origin: %s", str(e))
            
    if not repo_full_name:
        log('WARNING', "-> PR skipped: GITHUB_REPO/GITHUB_REPOSITORY is missing and couldn't be detected from git")
        return {"pr_created": False, "reason": "missing-GITHUB_REPO"}
    branch_name = f"agentic/auto-fix/iter_{iteration}"
    commit_msg = f"agentic: apply auto-fix (iter {iteration})"
    repo_file_path = os.getenv("REPO_FILE_PATH") or state.get("current_file") or staged_path.name
    try:
        log('INFO', "-> Creating branch %s and committing...", branch_name)
        create_branch_and_commit(repo_full_name, branch_name, repo_file_path, content, commit_msg)
        rca_path = AGENTIC_TMP_DIR / f"iter_{iteration}_rca.json"
        rca_text = rca_path.read_text() if rca_path.exists() else "RCA not found"
        pr_title = f"agentic: auto-fix (iter {iteration})"
        pr_body = f"Automated fix from agentic (iteration {iteration})\n\nRCA:\n```\n{rca_text}\n```"
        log('INFO', "-> Opening PR...")
        pr_url = open_pr_with_rca(repo_full_name, branch_name, pr_title, pr_body)
        log('INFO', "-> PR successfully created: %s", pr_url)
        return {"pr_created": True, "pr_url": pr_url}
    except Exception as e:
        log('ERROR', "-> Failed to create PR: %s", str(e))
        return {"pr_created": False, "reason": str(e)}


# ==========================================
# THE STATE 
# ==========================================
class AgenticState(TypedDict):
    logs: str                  
    files_to_fix: List[str]    
    current_file: str          
    patched_code: str          
    patched_file_path: Optional[str]
    iteration_count: int       
    success: bool                
    additional_info: Optional[str]   
    approved: Optional[bool]
    rca_path: Optional[str]

# ==========================================
# THE NODES
# ==========================================
def fetch_logs_node(state: AgenticState) -> dict:    
    log('INFO', "NODE[fetch_logs_node]: Fetching logs...")

    if state.get("iteration_count", 0) > 0:
        log('INFO', "-> Fetching latest test logs from the secure Sandbox.")
        return {"logs": state["logs"]}

    # Try to fetch real logs from the triggering GitHub Actions workflow run
    workflow_run_id = os.getenv("workflow_run_id")
    repo_full_name = os.getenv("GITHUB_REPO") or os.getenv("GITHUB_REPOSITORY")

    if workflow_run_id and repo_full_name:
        log('INFO', "-> Fetching REAL crash logs from GitHub Actions (run_id=%s)...", workflow_run_id)
        try:
            gh = github_client()
            repo = gh.get_repo(repo_full_name)
            run = repo.get_workflow_run(int(workflow_run_id))
            # Download logs from all failed jobs
            logs_parts = []
            for job in run.jobs():
                if job.conclusion == "failure":
                    log('INFO', "-> Fetching logs for failed job: %s", job.name)
                    # Get the log text for each failed step
                    for step in job.steps:
                        if step.conclusion == "failure":
                            logs_parts.append(
                                f"=== Job: {job.name} | Step: {step.name} ===\n"
                                f"Status: {step.conclusion}\n"
                            )
            # Also get the full run logs (downloadable zip → text)
            import requests as req
            import zipfile
            import io
            token = os.getenv("GITHUB_TOKEN")
            headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3+json"}
            logs_url = f"https://api.github.com/repos/{repo_full_name}/actions/runs/{workflow_run_id}/logs"
            resp = req.get(logs_url, headers=headers, allow_redirects=True)
            if resp.status_code == 200:
                z = zipfile.ZipFile(io.BytesIO(resp.content))
                for name in z.namelist():
                    content = z.read(name).decode("utf-8", errors="replace")
                    # Only include test/run step logs, skip setup noise
                    if any(kw in name.lower() for kw in ["run tests", "test", "build", "run"]):
                        logs_parts.append(f"=== {name} ===\n{content[-3000:]}\n")
                    elif not logs_parts:
                        # If no keyword match, include last 2000 chars as fallback
                        logs_parts.append(f"=== {name} ===\n{content[-2000:]}\n")
            if logs_parts:
                real_logs = "\n".join(logs_parts)
                # Truncate to avoid exceeding LLM context
                if len(real_logs) > 8000:
                    real_logs = real_logs[-8000:]
                log('INFO', "-> Fetched %d chars of real logs from %d log sections", len(real_logs), len(logs_parts))
                return {"logs": real_logs}
            else:
                log('WARNING', "-> No failure logs found in workflow run, falling back to simulated logs.")
        except Exception as e:
            log('ERROR', "-> Failed to fetch real logs: %s. Falling back to simulated logs.", str(e))

    log('INFO', "-> Using simulated crash logs (no workflow_run_id or fetch failed).")
    simulated_logs = """Traceback (most recent call last):
  File "app.py", line 10, in <module>
    import requests
ModuleNotFoundError: No module named 'requests'"""

    return {"logs": simulated_logs}

def analyze_code_node(state: AgenticState) -> dict:
    log('INFO', "NODE[analyze_code_node]: Gemini is analyzing logs to determine root cause...")

    # Build a listing of repo files to help the LLM pick the correct filename
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
    Read the following error logs and identify which specific file needs to be fixed.

    AVAILABLE FILES IN REPO: {file_listing}

    ERROR LOGS:
    {state['logs']}

    You MUST choose the file_to_fix from the AVAILABLE FILES listed above.
    Return your analysis strictly as a JSON object with two keys:
    1. "file_to_fix": The exact file path from AVAILABLE FILES (e.g., "example.py")
    2. "context": A brief explanation of what is wrong.
    """

    # Use the safe invoke wrapper
    response = invoke_with_retries(prompt)
    resp_text = extract_text(response)
    save_audit(state.get('iteration_count', 0), 'analyze_response', prompt, resp_text)

    try:
        analysis = safe_load_json(resp_text)
    except Exception:
        log('ERROR', "Failed to parse analysis JSON from LLM")
        # fallback: conservative default
        return {"files_to_fix": [], "current_file": "", "additional_info": "failed to parse LLM analysis"}

    if not validate_analysis_schema(analysis):
        log('WARNING', "Analysis JSON did not match expected schema: %s", str(analysis))
        return {"files_to_fix": [], "current_file": "", "additional_info": "invalid analysis schema"}

    file_target = analysis["file_to_fix"]
    log('INFO', "-> Target Acquired: %s", file_target)
    log('INFO', "-> AI Diagnosis: %s", analysis["context"]) 

    return {
        "files_to_fix": [file_target],
        "current_file": file_target,
        "additional_info": analysis["context"]
    }

def fix_code_node(state: AgenticState) -> dict:
    current_file = state["current_file"]
    iteration = state.get("iteration_count", 0)
    log('INFO', "NODE[fix_code_node]: Gemini is generating patch for %s (iter=%d)", current_file, iteration)

    try:
        broken_code = Path(current_file).read_text()
    except Exception:
        broken_code = "# File was missing or path is incorrect."
        log('WARNING', "-> Warning: %s not found locally. Proceeding with empty context.", current_file)

    prompt = f"""
    Fix the bug in the following python code.

    ERROR LOGS: {state.get('logs','')}
    AI CONTEXT: {state.get('additional_info', 'None')}

    CURRENT CODE:
    {broken_code}

    Rewrite the entire file to fix the bug.
    RETURN ONLY THE RAW PYTHON CODE. Do not include markdown formatting like ```python.
    """

    response = invoke_with_retries(prompt)
    resp_text = extract_text(response)
    # strip fences if present and save audit
    cleaned = strip_fences(resp_text)
    save_audit(iteration, 'fix_response', prompt, resp_text)

    # Stage patch locally (always). PR creation is gated by 'create_pr' in ALLOWED_ACTIONS.
    if 'create_pr' not in ALLOWED_ACTIONS:
        log('INFO', "create_pr not enabled; patch will be staged locally only.")

    patched_path = write_temp_patch(iteration + 1, current_file, cleaned)
    new_iteration_count = iteration + 1

    log('INFO', "-> Patch generated and staged at %s (Iteration %d)", str(patched_path), new_iteration_count)

    return {
        "patched_code": cleaned,
        "patched_file_path": str(patched_path),
        "iteration_count": new_iteration_count
    }

def test_code_node(state: AgenticState) -> dict:
    log('INFO', "NODE[test_code_node]: Injecting patch into Docker Sandbox...")

    sandbox_result = run_tests(
        project_path=".",
        test_command="python -m unittest discover",
        test_code_files=[],
        patched_code_string=state.get("patched_code", ""),
        file_to_fix=state.get("current_file")
    )

    if sandbox_result.get("success"):
        log('INFO', "-> Sandbox Execution: SUCCESS")
    else:
        log('WARNING', "-> Sandbox Execution: FAILED. Gathering new logs...")

    result = {
        "success": sandbox_result.get("success", False),
        "logs": sandbox_result.get("logs", "")
    }

    # If success, generate Root Cause Analysis (RCA) and save artifact
    if result["success"]:
        iteration = state.get('iteration_count', 0)
        rca_prompt = f"""
        You are a Senior Python Developer. Given the failure logs and the patch applied, produce a JSON object with keys:
        1. "summary": short summary of what was fixed
        2. "root_cause": root cause of the failure
        3. "patch_summary": short description of the patch

        FAILURE LOGS:
        {state.get('logs','')}

        PATCH (first 2000 chars):
        {state.get('patched_code','')[:2000]}
        """

        try:
            r_resp = invoke_with_retries(rca_prompt)
            r_text = extract_text(r_resp)
            save_audit(iteration, 'rca_response', rca_prompt, r_text)
            rca_obj = safe_load_json(r_text)
            if validate_rca_schema(rca_obj):
                rca_path = AGENTIC_TMP_DIR / f"iter_{iteration}_rca.json"
                rca_path.write_text(json.dumps(rca_obj, indent=2))
                result['rca_path'] = str(rca_path)
                log('INFO', "-> RCA saved to %s", str(rca_path))
            else:
                log('WARNING', "RCA did not match expected schema: %s", str(rca_obj))
        except Exception:
            log('ERROR', "Failed to generate or save RCA")

    return result
    
# ==========================================
# THE ROUTER (Traffic Cop)
# ==========================================
def route_after_test(state: AgenticState) -> str:
    if state.get("success"):
        log('INFO', "[ROUTER]: Tests passed! Moving to approval/PR stage.")
        return "wait_for_approval"
    elif state.get("iteration_count", 0) >= MAX_ITERATIONS:
        log('INFO', "[ROUTER]: Maximum iterations reached (%d). Ending process to prevent infinite loop.", MAX_ITERATIONS)
        return END
    else:
        log('INFO', "[ROUTER]: Tests failed. Looping back to analyzer with new Sandbox logs.")
        return "analyze_code"

# ==========================================
# BUILD THE GRAPH
# ==========================================
agent_builder = StateGraph(AgenticState)

agent_builder.add_node("fetch_logs", fetch_logs_node)
agent_builder.add_node("analyze_code", analyze_code_node) 
agent_builder.add_node("fix_code", fix_code_node)
agent_builder.add_node("test_code", test_code_node)
agent_builder.add_node("wait_for_approval", wait_for_approval_node)
agent_builder.add_node("create_pr", create_pr_node)

agent_builder.add_edge(START, "fetch_logs")
agent_builder.add_edge("fetch_logs", "analyze_code")
agent_builder.add_edge("analyze_code", "fix_code")
agent_builder.add_edge("fix_code", "test_code")

agent_builder.add_conditional_edges("test_code", route_after_test)

# Wire HITL/PR transitions using the StateGraph conditional API
def route_after_approval(state: AgenticState) -> str:
    return "create_pr" if state.get("approved", False) else END

agent_builder.add_conditional_edges("wait_for_approval", route_after_approval)
agent_builder.add_edge("create_pr", END)

agentic_graph = agent_builder.compile()

# ==========================================
# EXECUTION TRIGGER 
# ==========================================
if __name__ == "__main__":
    log('INFO', "🚀 Starting Pipeline Doctor Agent (powered by LLM)...")

    initial_state = {
        "iteration_count": 0,
        "success": False,
        "logs": ""
    }

    for step in agentic_graph.stream(initial_state):
        pass