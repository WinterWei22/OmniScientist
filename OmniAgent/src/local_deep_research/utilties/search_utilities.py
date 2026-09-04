import logging
import re
import asyncio
import fcntl
import json


def write_log_process_safe(log_path, message):
    try:
        with open(log_path, "a", encoding="utf-8") as log_file:
            fcntl.flock(log_file, fcntl.LOCK_EX)
            log_file.write(message)
            log_file.flush()
            fcntl.flock(log_file, fcntl.LOCK_UN)
    except Exception as log_exc:
        print(f"Failed to write to log file: {log_exc}")


def write_json_log_process_safe(log_path, new_dict):
    """
    Write a new dictionary to a JSON log file. The file content is a list of dictionaries.
    
    Args:
        log_path: JSON log file path
        new_dict: The new dictionary to be appended
    """
    try:
        data = []
        with open(log_path, "a+", encoding="utf-8") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                f.seek(0)
                content = f.read()
                if content:
                    data = json.loads(content)
                
                data.append(new_dict)
                
                # Clear file and write
                f.seek(0)
                f.truncate()
                json.dump(data, f, ensure_ascii=False, indent=2)
                f.flush()
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
    except Exception as e:
        print(f"Failed to write to JSON log file: {e}")
        


async def invoke_with_timeout_and_retry(llm, messages, timeout=90.0, max_retries=3, retry_delay=60.0):
    for attempt in range(max_retries):
        try:
            response = await asyncio.wait_for(
                llm.ainvoke(messages),
                timeout=timeout
            )
            return response
        except asyncio.TimeoutError:
            print(f"Attempt {attempt + 1}/{max_retries} timed out after {timeout}s")
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay}s...")
                await asyncio.sleep(retry_delay)
        except Exception as e:
            print(f"Attempt {attempt + 1}/{max_retries} failed with error: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay}s...")
                await asyncio.sleep(retry_delay)
            else:
                logging.error("Max retries reached. Giving up.")
                raise Exception(f"Failed after {max_retries} attempts: {e}")
    raise Exception("Failed to get a response after multiple attempts.")


def remove_think_tags(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return text


def extract_links_from_search_results(search_results: list) -> list:
    """
    Extracts links and titles from a list of search result dictionaries.

    Each dictionary is expected to have at least the keys "title" and "link".

    Returns a list of dictionaries with 'title' and 'url' keys.
    """
    links = []
    for result in search_results:
        try:
            title = result.get("title", "").strip()
            url = result.get("link", "").strip()
            index = result.get("index", "")
            if type(index) == str:
                index = index.strip()

            if title and url:
                links.append({"title": title, "url": url, "index": index})
        except Exception:
            continue
    return links


def format_links(links):
    formatted_links = ""
    formatted_links += "SOURCES:\n"
    for i, link in enumerate(links, 1):
        formatted_links += f"{link['index']}. {link['title']}\n   URL: {link['url']}\n"
    formatted_links += "\n"
    return formatted_links


def format_findings_to_text(findings_list, current_knowledge, questions_by_iteration):
    formatted_text = "COMPLETE RESEARCH OUTPUT \n\n"

    # Store the full current knowledge

    formatted_text += f"{current_knowledge}\n\n"
    formatted_text += "=" * 80 + "\n\n"

    # Store questions by iteration
    formatted_text += "SEARCH QUESTIONS BY ITERATION:\n"
    for iter_num, questions in questions_by_iteration.items():
        formatted_text += f"\nIteration {iter_num}:\n"
        for i, q in enumerate(questions, 1):
            formatted_text += f"{i}. {q}\n"
    formatted_text += "\n" + "=" * 80 + "\n\n"

    # Store detailed findings
    formatted_text += "DETAILED FINDINGS:\n\n"
    all_links = []  # To collect all sources

    for finding in findings_list:
        # Phase header
        formatted_text += f"{'=' * 80}\n"
        formatted_text += f"PHASE: {finding['phase']}\n"
        formatted_text += f"{'=' * 80}\n\n"

        # If this is a follow-up phase, show the corresponding question
        if finding["phase"].startswith("Follow-up"):
            iteration = int(finding["phase"].split(".")[0].split()[-1])
            question_index = int(finding["phase"].split(".")[-1]) - 1
            if iteration in questions_by_iteration and question_index < len(
                questions_by_iteration[iteration]
            ):
                formatted_text += f"SEARCH QUESTION:\n{questions_by_iteration[iteration][question_index]}\n\n"

        # Content
        formatted_text += f"CONTENT:\n{finding['content']}\n\n"

        # Search results if they exist
        if "search_results" in finding:
            # formatted_text += "SEARCH RESULTS:\n"
            # formatted_text += f"{finding['search_results']}\n\n"

            # Extract and format links for this finding
            links = extract_links_from_search_results(finding["search_results"])
            if links:
                formatted_text += "SOURCES USED IN THIS SECTION:\n"
                for i, link in enumerate(links, 1):
                    formatted_text += f"{i}. {link['title']}\n   URL: {link['url']}\n"
                formatted_text += "\n"
                all_links.extend(links)

        formatted_text += f"{'_' * 80}\n\n"

    # Add summary of all sources at the end
    if all_links:
        formatted_text += "\nALL SOURCES USED IN RESEARCH:\n"
        formatted_text += "=" * 80 + "\n\n"
        seen_urls = set()  # To prevent duplicates
        for i, link in enumerate(all_links, 1):
            if link["url"] not in seen_urls:
                formatted_text += f"{i}. {link['title']}\n   URL: {link['url']}\n"
                seen_urls.add(link["url"])
        formatted_text += "\n" + "=" * 80 + "\n"

    return formatted_text


def print_search_results(search_results):
    formatted_text = ""
    links = extract_links_from_search_results(search_results)
    if links:
        formatted_text = format_links(links=links)
    logging.info(formatted_text)
