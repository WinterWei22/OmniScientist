import os
import json
from datetime import datetime
from typing import Any, Dict, List

class ResearchTraceLogger:
    def __init__(self, log_dir: str, query: str, question_id: str | int = "unknown"):
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
            
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        safe_id = str(question_id).replace("/", "_").replace("\\", "_")
        
        # 1. Full Trace
        self.full_log_path = os.path.join(log_dir, f"trace_{safe_id}_full.md")
        
        # 2. Clean Trace
        self.clean_log_path = os.path.join(log_dir, f"trace_{safe_id}_clean.md")
        
        # 3. Case Log
        self.case_json_path = os.path.join(log_dir, f"case_{safe_id}.json")
        
        self.query = query
        self.question_id = question_id
        
        self.case_data = {
            "question_id": question_id,
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "template": None,
            "steps": [],
            "final_answer": None,
            "final_report": None,
            "tools": [] 
        }
        
        self._init_logs()

    def _init_logs(self):
        header = f"# 🧬 OmniAgent Research Trace (ID: {self.question_id})\n\n"
        header += f"**Topic**: `{self.query}`\n"
        header += f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        header += "---\n\n"

        with open(self.full_log_path, "w", encoding="utf-8") as f:
            f.write(header)
            
        with open(self.clean_log_path, "w", encoding="utf-8") as f:
            f.write(header)

    def log_template(self, template_content: str):
        self.case_data["template"] = template_content
        
        md_content = f"\n## 🧠 Thinking Template\n\n<details><summary><b>View Template</b></summary>\n\n```text\n{template_content}\n```\n\n</details>\n\n"
        
        with open(self.full_log_path, "a", encoding="utf-8") as f:
            f.write(md_content)
            
        with open(self.clean_log_path, "a", encoding="utf-8") as f:
            f.write(md_content)

    def log_phase(self, phase_name: str, iteration: int = None):
        timestamp = datetime.now().strftime("%H:%M:%S")
        md_content = ""
        
        if iteration is not None:
            md_content = f"\n\n# 🔄 Iteration {iteration}: {phase_name}\n**Time**: {timestamp}\n\n"
        else:
            md_content = f"\n\n# 📑 {phase_name}\n**Time**: {timestamp}\n\n"

        with open(self.full_log_path, "a", encoding="utf-8") as f:
            f.write(md_content)
        with open(self.clean_log_path, "a", encoding="utf-8") as f:
            f.write(md_content)
            
        self.case_data["steps"].append({
            "type": "phase",
            "name": phase_name,
            "iteration": iteration,
            "timestamp": timestamp
        })

    def log_agent_activity(self, agent_name: str, input_prompt: str, output_content: Any, thoughts: str = None):
        md_full = self._format_agent_activity_md(agent_name, input_prompt, output_content, thoughts, truncate=False)
        
        md_clean = self._format_agent_activity_md(agent_name, input_prompt, output_content, thoughts, truncate=False)
        
        with open(self.full_log_path, "a", encoding="utf-8") as f:
            f.write(md_full)
        with open(self.clean_log_path, "a", encoding="utf-8") as f:
            f.write(md_clean)

        self.case_data["steps"].append({
            "type": "agent",
            "role": agent_name,
            "prompt": input_prompt, 
            "output": output_content,
            "thoughts": thoughts
        })
        
        if "Final Answer" in agent_name or "Writer" in agent_name:
             if isinstance(output_content, str):
                 self.case_data["final_answer"] = output_content

    def _format_agent_activity_md(self, agent_name, input_prompt, output_content, thoughts, truncate=False):
        content = f"### 🤖 Agent: {agent_name}\n\n"
        
        if thoughts:
            formatted_thoughts = thoughts.replace("\n", "\n> ")
            content += f"**💭 Thoughts:**\n\n> {formatted_thoughts}\n\n"
        
        content += f"<details><summary><b>View Input Prompt</b></summary>\n\n```text\n{input_prompt}\n```\n\n</details>\n\n"
        
        content += f"**📝 Output:**\n\n"
        
        formatted_output = ""
        if isinstance(output_content, (dict, list)):
             formatted_output = json.dumps(output_content, indent=2, ensure_ascii=False)
        else:
             formatted_output = str(output_content)

        if truncate and len(formatted_output) > 200:
             content += f"{formatted_output[:200]}... [TRUNCATED]\n\n"
        else:
             if isinstance(output_content, (dict, list)):
                 content += f"```json\n{formatted_output}\n```\n\n"
             else:
                 content += f"{formatted_output}\n\n"
                 
        content += "---\n"
        return content

    def log_sub_queries(self, queries: List[str]):
        md_content = f"#### 🔍 Generated Sub-queries\n\n"
        for i, q in enumerate(queries, 1):
            md_content += f"{i}. {q}\n"
        md_content += "\n"

        with open(self.full_log_path, "a", encoding="utf-8") as f:
            f.write(md_content)
        with open(self.clean_log_path, "a", encoding="utf-8") as f:
            f.write(md_content)
            
        self.case_data["steps"].append({
            "type": "sub_queries",
            "queries": queries
        })

    def log_tool_execution(self, tool_name: str, tool_input: Dict, tool_output: str, success: bool):
        self._write_tool_log(self.full_log_path, tool_name, tool_input, tool_output, success, truncate=False)
        
        self._write_tool_log(self.clean_log_path, tool_name, tool_input, tool_output, success, truncate=True)
        
        self._record_tool_json(tool_name, tool_input, tool_output, success, truncate=True)

    def _write_tool_log(self, path, tool_name, tool_input, tool_output, success, truncate=False):
        icon = "✅" if success else "❌"
        status = "Success" if success else "Failed"
        
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"#### 🛠️ Tool Call: `{tool_name}` {icon}\n\n")
            f.write(f"- **Status**: {status}\n")
            
            input_str = json.dumps(tool_input, ensure_ascii=False)
            if truncate and len(input_str) > 200:
                f.write(f"- **Input**: `{input_str[:200]}...`\n\n")
            else:
                f.write(f"- **Input**: `{input_str}`\n\n")
            
            try:
                if isinstance(tool_output, str):
                    parsed = json.loads(tool_output)
                    formatted_output = json.dumps(parsed, indent=2, ensure_ascii=False)
                else:
                    formatted_output = json.dumps(tool_output, indent=2, ensure_ascii=False)
            except:
                formatted_output = str(tool_output)

            if truncate:
                 preview = formatted_output[:200].replace("\n", " ") + "..." if len(formatted_output) > 200 else formatted_output
                 f.write(f"**Output (Truncated)**: `{preview}`\n\n")
            else:
                 f.write(f"<details><summary><b>Full Output Content</b></summary>\n\n```json\n{formatted_output}\n```\n\n</details>\n\n")

    def _record_tool_json(self, tool_name, tool_input, tool_output, success, truncate=True):
        input_data = tool_input
        output_data = tool_output

        if truncate:
             input_str = str(tool_input)
             if len(input_str) > 200:
                 input_data = input_str[:200] + "... [TRUNCATED]"
                 
             output_str = str(tool_output)
             if len(output_str) > 200:
                 output_data = output_str[:200] + "... [TRUNCATED]"

        self.case_data["steps"].append({
            "type": "tool",
            "name": tool_name,
            "input": input_data,
            "output": output_data,
            "success": success
        })
        
        self.case_data["tools"].append({
            "name": tool_name,
            "input": input_data,
            "output_preview": output_data if isinstance(output_data, str) else str(output_data),
            "success": success
        })

    def log_knowledge_update(self, new_knowledge: str, priority: int = 1):
        with open(self.full_log_path, "a", encoding="utf-8") as f:
            f.write(f"#### 📚 Knowledge Added (Priority {priority})\n\n")
            f.write(f"```markdown\n{new_knowledge}\n```\n\n")

        truncated_knowledge = new_knowledge
        if len(new_knowledge) > 200:
            truncated_knowledge = new_knowledge[:200] + "... [TRUNCATED]"

        self.case_data["steps"].append({
            "type": "knowledge",
            "priority": priority,
            "content": truncated_knowledge
        })

    def log_error(self, error_msg: str):
        content = f"### ⚠️ Error\n\n> {error_msg}\n\n"
        with open(self.full_log_path, "a", encoding="utf-8") as f:
            f.write(content)
        with open(self.clean_log_path, "a", encoding="utf-8") as f:
            f.write(content)
            
        self.case_data["steps"].append({
            "type": "error",
            "message": error_msg
        })

    def save_case_json(self):
        with open(self.case_json_path, "w", encoding="utf-8") as f:
            json.dump(self.case_data, f, indent=2, ensure_ascii=False)
