from __future__ import annotations

import json
import unittest

from openai_bridge import build_prompt, parse_tool_calls


class PromptTests(unittest.TestCase):
    def test_responses_prompt_keeps_tool_result_and_schema(self) -> None:
        prompt = build_prompt(
            {
                "instructions": "Be concise.",
                "input": [
                    {
                        "role": "user",
                        "content": [{"type": "input_text", "text": "Read file."}],
                    },
                    {
                        "type": "function_call_output",
                        "call_id": "call_1",
                        "output": "file contents",
                    },
                ],
                "tools": [
                    {
                        "type": "function",
                        "name": "read_file",
                        "description": "Read one file",
                        "parameters": {
                            "type": "object",
                            "properties": {"path": {"type": "string"}},
                            "required": ["path"],
                        },
                    }
                ],
            },
            "responses",
        )
        self.assertIn("Be concise.", prompt)
        self.assertIn("TOOL RESULT (call_id=call_1)", prompt)
        self.assertIn("file contents", prompt)
        self.assertIn('"name": "read_file"', prompt)

    def test_chat_prompt_keeps_roles(self) -> None:
        prompt = build_prompt(
            {
                "messages": [
                    {"role": "system", "content": "Be exact."},
                    {"role": "user", "content": "2+2?"},
                ]
            },
            "chat",
        )
        self.assertIn("SYSTEM:\nBe exact.", prompt)
        self.assertIn("USER:\n2+2?", prompt)


class ToolCallTests(unittest.TestCase):
    def test_parses_bridge_tool_calls(self) -> None:
        calls = parse_tool_calls(
            json.dumps(
                {
                    "type": "tool_calls",
                    "calls": [
                        {
                            "name": "read_file",
                            "arguments": {"path": "README.md"},
                        }
                    ],
                }
            ),
            {"read_file"},
        )
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "read_file")
        self.assertEqual(
            json.loads(calls[0].arguments),
            {"path": "README.md"},
        )

    def test_does_not_steal_ordinary_json_answer(self) -> None:
        calls = parse_tool_calls('{"answer": 4}', {"read_file"})
        self.assertEqual(calls, ())

    def test_rejects_unknown_tool(self) -> None:
        calls = parse_tool_calls(
            '{"type":"tool_calls","calls":[{"name":"delete_world","arguments":{}}]}',
            {"read_file"},
        )
        self.assertEqual(calls, ())

    def test_repairs_raw_newline_in_fetch_webpage_query(self) -> None:
        calls = parse_tool_calls(
            '''{"type":"tool_calls","calls":[{"name":"fetch_webpage","arguments":{"urls":["https://github.com/go-rod/rod","https://github.com/cdpdriver/zendriver"],"query":"Compare
go-rod/rod and cdpdriver/zendriver as browser automation options"}}]}''',
            {"fetch_webpage"},
        )

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "fetch_webpage")

        arguments = json.loads(calls[0].arguments)
        self.assertEqual(
            arguments["query"],
            "Compare\ngo-rod/rod and cdpdriver/zendriver as browser automation options",
        )
        self.assertEqual(
            arguments["urls"],
            [
                "https://github.com/go-rod/rod",
                "https://github.com/cdpdriver/zendriver",
            ],
        )

    def test_repairs_unescaped_windows_path_and_ignores_trailing_markdown(
        self,
    ) -> None:
        calls = parse_tool_calls(
            r'''{"type":"tool_calls","calls":[{"name":"apply_patch","arguments":{"input":"*** Begin Patch\n*** Update File: c:\Users\Mihir\Code\WebLLM2API\test.txt\n-this is a test\n+this is a test\n*** End Patch","explanation":"Repeat the line."}}]}******''',
            {"apply_patch"},
        )
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "apply_patch")
        arguments = json.loads(calls[0].arguments)
        self.assertIn(
            r"c:\Users\Mihir\Code\WebLLM2API\test.txt",
            arguments["input"],
        )
        self.assertNotIn("\t", arguments["input"])

    def test_prompt_warns_to_escape_windows_paths(self) -> None:
        prompt = build_prompt(
            {
                "input": "Edit a file.",
                "tools": [
                    {
                        "type": "function",
                        "name": "apply_patch",
                        "parameters": {"type": "object"},
                    }
                ],
            },
            "responses",
        )
        self.assertIn(r"C:\\Users\\name", prompt)

    def test_repairs_control_escape_in_direct_windows_path(self) -> None:
        calls = parse_tool_calls(
            r'''{"type":"tool_calls","calls":[{"name":"inspect_path","arguments":{"path":"C:\Users\Mihir\Code\WebLLM2API\test.txt"}}]}''',
            {"inspect_path"},
        )
        self.assertEqual(len(calls), 1)
        arguments = json.loads(calls[0].arguments)
        self.assertEqual(
            arguments["path"],
            r"C:\Users\Mihir\Code\WebLLM2API\test.txt",
        )
        self.assertNotIn("\t", arguments["path"])

    def test_repairs_valid_json_tab_escape_in_mixed_windows_path(self) -> None:
        calls = parse_tool_calls(
            r'''{"type":"tool_calls","calls":[{"name":"inspect_path","arguments":{"path":"C:\\Users\\Mihir\\Code\\WebLLM2API\test.txt"}}]}''',
            {"inspect_path"},
        )
        arguments = json.loads(calls[0].arguments)
        self.assertEqual(
            arguments["path"],
            r"C:\Users\Mihir\Code\WebLLM2API\test.txt",
        )

    def test_repairs_regex_quotes_and_preserves_parallel_calls(self) -> None:
        calls = parse_tool_calls(
            r'''{"type":"tool_calls","calls":[{"name":"read_file","arguments":{"filePath":"c:\Users\Mihir\Code\WebLLM2API\openai_bridge.py","startLine":401,"endLine":900}},{"name":"run_in_terminal","arguments":{"command":"uv run python -m unittest -q","mode":"sync"}},{"name":"grep_search","arguments":{"query":"TODO|FIXME|allow_origins=\["\"\]|print\(","isRegexp":true,"includePattern":"**/.py","maxResults":100}}]}''',
            {"read_file", "run_in_terminal", "grep_search"},
        )
        self.assertEqual(
            [call.name for call in calls],
            ["read_file", "run_in_terminal", "grep_search"],
        )
        read_arguments = json.loads(calls[0].arguments)
        self.assertEqual(
            read_arguments["filePath"],
            r"c:\Users\Mihir\Code\WebLLM2API\openai_bridge.py",
        )
        grep_arguments = json.loads(calls[2].arguments)
        self.assertEqual(
            grep_arguments["query"],
            r'''TODO|FIXME|allow_origins=\[""\]|print\(''',
        )

    def test_salvages_valid_parallel_call_when_another_is_irreparable(
        self,
    ) -> None:
        calls = parse_tool_calls(
            r'''{"type":"tool_calls","calls":[{"name":"read_file","arguments":{"filePath":"README.md","startLine":1,"endLine":20}},{"name":"grep_search","arguments":not-json}]}''',
            {"read_file", "grep_search"},
        )
        self.assertEqual([call.name for call in calls], ["read_file"])


if __name__ == "__main__":
    unittest.main()
