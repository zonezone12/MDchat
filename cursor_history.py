#!/usr/bin/env python3
import argparse
import base64
import json
import os
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from urllib.parse import unquote, urlparse


def cursor_state_vscdb_path() -> Path:
    """Path to Cursor's global state DB (platform-specific)."""
    home = Path.home()
    if sys.platform == "win32":
        appdata = os.environ.get("APPDATA")
        if not appdata:
            raise RuntimeError("APPDATA is not set; cannot locate Cursor data on Windows")
        base = Path(appdata) / "Cursor" / "User" / "globalStorage"
    elif sys.platform == "darwin":
        base = home / "Library" / "Application Support" / "Cursor" / "User" / "globalStorage"
    else:
        base = home / ".config" / "Cursor" / "User" / "globalStorage"
    return base / "state.vscdb"


def cursor_workspace_storage_root() -> Path:
    """Cursor per-workspace metadata (workspace.json, etc.)."""
    home = Path.home()
    if sys.platform == "win32":
        appdata = os.environ.get("APPDATA")
        if not appdata:
            raise RuntimeError("APPDATA is not set; cannot locate workspaceStorage on Windows")
        return Path(appdata) / "Cursor" / "User" / "workspaceStorage"
    if sys.platform == "darwin":
        return home / "Library" / "Application Support" / "Cursor" / "User" / "workspaceStorage"
    return home / ".config" / "Cursor" / "User" / "workspaceStorage"


def file_uri_to_path(uri: str) -> Path | None:
    """Turn a workspace.json style file URI into a local Path."""
    parsed = urlparse(uri)
    if parsed.scheme != "file":
        return None
    path = unquote(parsed.path)
    if len(path) >= 3 and path[0] == "/" and path[2] == ":":
        path = path[1:]
    return Path(path)


def find_workspace_storage_id(project_root: Path) -> str | None:
    """
    Find the workspaceStorage folder name (hash) for a single-folder workspace,
    by scanning workspace.json entries under workspaceStorage.
    """
    root = project_root.resolve()
    ws_root = cursor_workspace_storage_root()
    if not ws_root.is_dir():
        return None
    for entry in ws_root.iterdir():
        if not entry.is_dir():
            continue
        wj = entry / "workspace.json"
        if not wj.is_file():
            continue
        try:
            data = json.loads(wj.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        folder = data.get("folder")
        if not folder or not isinstance(folder, str):
            continue
        p = file_uri_to_path(folder)
        if p is None:
            continue
        try:
            if p.resolve() == root:
                return entry.name
        except OSError:
            if os.path.normcase(os.path.normpath(str(p))) == os.path.normcase(
                os.path.normpath(str(root))
            ):
                return entry.name
    return None


def path_is_under_project(path: Path, project_root: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(project_root.resolve())
        return True
    except ValueError:
        return False
    except OSError:
        cr = os.path.normcase(os.path.normpath(str(path)))
        rr = os.path.normcase(os.path.normpath(str(project_root.resolve())))
        return cr == rr or cr.startswith(rr + os.sep)


def composer_ids_for_project(db_path: Path, project_root: Path) -> set[str]:
    """
    Composer (chat session) IDs that touched files under project_root, inferred from
    cursorDiskKV keys: ofsContent:<composerId>:<file-uri>.
    """
    project_root = project_root.resolve()
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT key FROM cursorDiskKV WHERE key LIKE 'ofsContent:%'"
        )
        ids: set[str] = set()
        for (key,) in cur.fetchall():
            rest = key[len("ofsContent:") :]
            colon = rest.find(":")
            if colon <= 0:
                continue
            composer_id = rest[:colon].strip().lower()
            uri = rest[colon + 1 :]
            p = file_uri_to_path(uri)
            if p is None:
                continue
            if path_is_under_project(p, project_root):
                ids.add(composer_id)
        return ids
    finally:
        conn.close()


def extract_conversations_from_db(db_path, composer_ids: set[str] | None = None):
    """Extract all conversations from the database with detailed error handling."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        
        # Get all bubbleId entries which contain the actual chat messages
        cursor.execute("SELECT key, value FROM cursorDiskKV WHERE key LIKE 'bubbleId:%'")
        bubble_rows = cursor.fetchall()
        print(f"Scanned {len(bubble_rows)} bubbleId rows in database")
        if composer_ids is not None:
            print(f"Filtering to {len(composer_ids)} composer id(s) tied to this folder (via ofsContent keys)")
        
        # Group messages by composer ID
        conversations_by_composer = {}
        
        for key, value in bubble_rows:
            if value is None:
                continue
                
            try:
                # Parse the key to extract composer ID and bubble ID
                # Format: bubbleId:composerId:bubbleId
                parts = key.split(':')
                if len(parts) >= 3:
                    composer_id = parts[1]
                    bubble_id = parts[2]
                    if composer_ids is not None and composer_id.lower() not in composer_ids:
                        continue
                    
                    # Decode the message data
                    try:
                        data = json.loads(value)
                    except json.JSONDecodeError:
                        try:
                            decoded = base64.b64decode(value)
                            data = json.loads(decoded)
                        except:
                            continue
                    
                    # Only include bubbles that have text content
                    if isinstance(data, dict) and 'text' in data and data['text'].strip():
                        # Initialize composer conversation if not exists
                        if composer_id not in conversations_by_composer:
                            conversations_by_composer[composer_id] = []
                        
                        # Add message to conversation
                        conversations_by_composer[composer_id].append({
                            'id': bubble_id,
                            'data': data,
                            'key': key
                        })
                    
            except Exception as e:
                print(f"Error processing bubble {key}: {e}")
        
        # Convert to conversations list
        conversations = []
        for composer_id, messages in conversations_by_composer.items():
            if messages:  # Only include composers with messages
                conversations.append({
                    'id': composer_id,
                    'messages': messages
                })
        
        print(f"Found {len(conversations)} conversations with messages")
        
        conn.close()
        return conversations
    except Exception as e:
        print(f"Error accessing database {db_path}: {e}")
        return []

def format_message(msg_data):
    """Format a single message with all its components."""
    if not isinstance(msg_data, dict) or 'data' not in msg_data:
        return "Invalid message format"
    
    msg = msg_data['data']
    formatted = []
    
    # Add timestamp if available
    timestamp_str = ""
    if 'timingInfo' in msg and 'clientStartTime' in msg['timingInfo']:
        timestamp = datetime.fromtimestamp(
            msg['timingInfo']['clientStartTime'] / 1000
        ).strftime("%Y-%m-%d %H:%M:%S")
        timestamp_str = f" ({timestamp})"
    
    # Determine message role based on type
    msg_type = msg.get('type', 0)
    if msg_type == 1:  # User message
        role = "User"
    elif msg_type == 2:  # Assistant message
        role = "Assistant"
    else:
        role = "Message"
    
    formatted.append(f"**{role}**{timestamp_str}:")
    
    # Add the main text content
    text = msg.get('text', '').strip()
    if text:
        formatted.append(text)
    
    # Add code blocks if present
    if 'codeBlocks' in msg and msg['codeBlocks']:
        for block in msg['codeBlocks']:
            if isinstance(block, dict):
                lang = block.get('language', '')
                code = block.get('code', '')
                if code:
                    formatted.append(f"```{lang}\n{code}\n```")
    
    # Add tool results if present
    if 'toolResults' in msg and msg['toolResults']:
        formatted.append("\n**Tool Results:**")
        for result in msg['toolResults']:
            if isinstance(result, dict):
                tool_name = result.get('toolName', 'Unknown Tool')
                formatted.append(f"- {tool_name}")
                if 'result' in result:
                    formatted.append(f"  Result: {result['result']}")
    
    return "\n\n".join(formatted) if formatted else "Empty message"

def main():
    """Main function to export all chat history."""
    parser = argparse.ArgumentParser(
        description="Export Cursor chat bubbles from global state.vscdb to markdown."
    )
    parser.add_argument(
        "--workspace-only",
        action="store_true",
        help="Only chats whose composer touched files under the project (ofsContent keys). "
        "Uses workspaceStorage to confirm the folder mapping.",
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=None,
        help="Project root directory (default: current working directory). Used with --workspace-only.",
    )
    args = parser.parse_args()

    global_db = cursor_state_vscdb_path()
    composer_ids: set[str] | None = None
    project_root: Path | None = None

    if args.workspace_only:
        project_root = (args.project or Path.cwd()).resolve()
        ws_id = find_workspace_storage_id(project_root)
        ws_root = cursor_workspace_storage_root()
        print(f"workspaceStorage: {ws_root}")
        if ws_id:
            print(f"Matched workspace folder: {ws_id} -> {project_root}")
        else:
            print(
                f"Warning: no workspace.json under workspaceStorage points to {project_root}; "
                "still filtering by ofsContent file paths under this folder."
            )
        composer_ids = composer_ids_for_project(global_db, project_root)
        print(f"Inferred {len(composer_ids)} composer session(s) from file activity under project")

    print("Extracting conversations from database...")

    print(f"Extracting conversations from {global_db}")
    conversations = extract_conversations_from_db(global_db, composer_ids=composer_ids)
    
    # Sort conversations by first message timestamp (if available)
    def get_conversation_timestamp(conv):
        if conv['messages']:
            # Try to find a timestamp in the first message
            first_msg = conv['messages'][0]['data']
            if isinstance(first_msg, dict) and 'timingInfo' in first_msg:
                timing = first_msg['timingInfo']
                if 'clientStartTime' in timing:
                    return timing['clientStartTime']
        return 0
    
    conversations.sort(key=get_conversation_timestamp)

    print("Generating markdown...")
    
    title = "# Cursor Chat History\n"
    if project_root is not None:
        title = f"# Cursor Chat History (workspace: `{project_root}`)\n"
    output = [title]
    for conv in conversations:
        output.append(f"## Conversation {conv['id']}\n")
        output.append(f"*Messages: {len(conv['messages'])}*\n")
        
        # Sort messages within conversation by timestamp
        def get_message_timestamp(msg):
            if 'data' in msg and 'timingInfo' in msg['data']:
                timing = msg['data']['timingInfo']
                if 'clientStartTime' in timing:
                    return timing['clientStartTime']
            return 0
        
        sorted_messages = sorted(conv['messages'], key=get_message_timestamp)
        
        for msg in sorted_messages:
            output.append(format_message(msg))
            output.append("\n---\n")
        
        output.append("\n")
    
    # Write to file
    output_dir = Path("chat_history_exports")
    output_dir.mkdir(exist_ok=True)
    suffix = "_workspace" if args.workspace_only else ""
    output_file = (
        output_dir
        / f"cursor_chat_history{suffix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    )
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(output))
    
    print(f"Exported {len(conversations)} conversations to {output_file}")

if __name__ == "__main__":
    main()