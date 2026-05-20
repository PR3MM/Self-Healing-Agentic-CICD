from __future__ import annotations
import os
import json
import time
import re
import ast
from pathlib import Path
from typing import List, Dict, Any, TypedDict, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from github import Github
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
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

class AnalyzeOutput(BaseModel):
    failure_summary: str = Field(description="Short description of the error")
    root_cause: str = Field(description="Detailed explanation of why it failed")
    target_file: str = Field(description="The EXACT file path from AVAILABLE FILES to fix")
    repair_strategy: str = Field(description="Step-by-step plan to fix it")
    confidence: float = Field(description="0.0 to 1.0 confidence score")

class EditChunk(BaseModel):
    start_line: int = Field(description="1-indexed start line")
    end_line: int = Field(description="1-indexed inclusive end line")
    replacement: str = Field(description="New code for these lines")

class FixOutput(BaseModel):
    edits: List[EditChunk] = Field(description="List of surgical edits")

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

    Analyze the logs and determine the root cause.
    """
    
    structured_llm = llm.with_structured_output(AnalyzeOutput).with_retry(stop_after_attempt=3)
    response = structured_llm.invoke(prompt)
    
    save_audit(state.get('iteration_count', 0), 'plan_response', prompt, str(response.dict()))

    try:
        target_file = response.target_file
        repair_strategy = response.repair_strategy
        
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

    Do NOT rewrite the whole file unless necessary.
    """

    structured_llm = llm.with_structured_output(FixOutput).with_retry(stop_after_attempt=3)
    response = structured_llm.invoke(prompt)
    
    save_audit(iteration, 'fix_response', prompt, str(response.dict()))

    try:
        edits = [e.dict() for e in response.edits]
        
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
        log('ERROR', "Failed to apply surgical edits: %s", str(e))

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
    <head>
        <title>Pipeline Doctor Report</title>
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@picocss/pico@1/css/pico.min.css">
        <style>body{{padding: 20px;}} .iter{{margin-top: 20px;}} .badge{{padding: 5px 10px; border-radius: 5px; color: white;}} .badge.Success{{background: #28a745;}} .badge.Failed{{background: #dc3545;}}</style>
    </head>
    <body>
    <main class="container">
    <h1>Pipeline Doctor - Repair Timeline</h1>
    <h3>Status: <span class="badge {rca_obj['final_status']}">{rca_obj['final_status']}</span></h3>
    <p><strong>Files Patched:</strong> {', '.join(rca_obj['files_modified'])}</p>
    """
    for it in iterations:
        result_badge = "Success" if it.get('result') == 'passed' else "Failed"
        html_content += f"""
        <article class='iter'>
            <header><h4>Iteration {it.get('iteration')} - {it.get('target_file')}</h4></header>
            <p><b>Strategy:</b> {it.get('strategy')}</p>
            <p><b>Result:</b> <span class="badge {result_badge}">{it.get('result')}</span> ({it.get('reason')})</p>
        </article>
        """
    html_content += "</main></body></html>"
    
    html_path = AGENTIC_TMP_DIR / "report.html"
    html_path.write_text(html_content)
    
    log('INFO', "-> RCA and HTML report generated.")
    return {"rca_path": str(rca_path), "rca_html_path": str(html_path)}

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
        
        iterations = state.get("repair_memory", {}).get("iterations", [])
        last_iter = iterations[-1] if iterations else {}
        target_file = last_iter.get("target_file", "multiple files")
        strategy = last_iter.get("strategy", "Automated code repair")

        pr_title = f"Pipeline Doctor: Auto-fix for {target_file}"
        pr_body = f"""## 🚨 Pipeline Doctor Automated Repair

**Root Cause & Strategy:**
{strategy}

**Files Modified:**
"""
        for f in repo_state.keys():
            pr_body += f"- `{f}`\n"
            
        pr_body += """
✅ **Sandbox Verification:**
The agent successfully tested this patch in an isolated Docker sandbox. All tests passed!

_Check the attached HTML report artifact for the full repair timeline and visualizations._
"""
        
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
    if state.get("success"):
        return "create_pr"
    return END

agent_builder = StateGraph(AgenticState)

agent_builder.add_node("fetch_logs", fetch_logs_node)
agent_builder.add_node("analyze_code", analyze_code_node) 
agent_builder.add_node("fix_code", fix_code_node)
agent_builder.add_node("lint_check", lint_check_node)
agent_builder.add_node("test_code", test_code_node)
agent_builder.add_node("generate_rca", generate_rca_node)
agent_builder.add_node("create_pr", create_pr_node)

agent_builder.add_edge(START, "fetch_logs")
agent_builder.add_edge("fetch_logs", "analyze_code")
agent_builder.add_edge("analyze_code", "fix_code")
agent_builder.add_edge("fix_code", "lint_check")

agent_builder.add_conditional_edges("lint_check", route_after_lint)
agent_builder.add_conditional_edges("test_code", route_after_test)
agent_builder.add_conditional_edges("generate_rca", route_after_rca)
agent_builder.add_edge("create_pr", END)

hitl = os.getenv("HITL_ENABLED", "true").lower() == "true"
memory = MemorySaver()
interrupts = ["create_pr"] if hitl else []
agentic_graph = agent_builder.compile(checkpointer=memory, interrupt_before=interrupts)

if __name__ == "__main__":
    # Generate Agentic Flow Visualization
    try:
        mermaid_graph = agentic_graph.get_graph().draw_mermaid()
        mermaid_path = AGENTIC_TMP_DIR / "agent_flow.mermaid"
        mermaid_path.write_text(mermaid_graph)
        log('INFO', f"Agent workflow visualization saved to {mermaid_path}")
    except Exception as e:
        log('WARNING', f"Could not generate mermaid graph: {e}")

    log('INFO', "🚀 Starting Pipeline Doctor Agent v2 (powered by LLM)...")
    initial_state = {"iteration_count": 0, "success": False, "logs": ""}
    thread_config = {"configurable": {"thread_id": "1"}}
    
    for step in agentic_graph.stream(initial_state, config=thread_config):
        pass

    # If HITL Breakpoint was hit, wait for manual marker file and resume
    if hitl:
        state = agentic_graph.get_state(thread_config)
        if state.next and "create_pr" in state.next:
            log('INFO', "⏸️ Graph execution paused for Human-in-the-Loop approval before creating PR.")
            timeout = int(os.getenv("HITL_TIMEOUT_SEC", "600"))
            interval = 5
            waited = 0
            approved = False
            iteration = state.values.get("iteration_count", 0)
            marker = AGENTIC_TMP_DIR / f"iter_{iteration}" / "approved"
            while waited < timeout:
                if marker.exists():
                    approved = True
                    break
                time.sleep(interval)
                waited += interval
            
            if approved:
                log('INFO', "▶️ Approval received. Resuming graph execution...")
                for step in agentic_graph.stream(None, config=thread_config):
                    pass
            else:
                log('INFO', "Approval timed out. PR not created.")