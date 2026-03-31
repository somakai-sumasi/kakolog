from unittest.mock import MagicMock, patch

from kakolog.chunker import TurnChunk
from kakolog.cleaner import clean_text, is_trivial
from kakolog.extractor import extract_conversations, extract_text
from kakolog.transcript import parse_jsonl


class TestCleanText:
    def test_removes_system_reminder(self):
        text = "hello <system-reminder>noise</system-reminder> world"
        assert clean_text(text) == "hello  world"

    def test_removes_noise_lines(self):
        text = "line1\nFull transcript available at: /foo\nline2"
        result = clean_text(text)
        assert "Full transcript" not in result
        assert "line1" in result
        assert "line2" in result

    def test_collapses_blank_lines(self):
        text = "a\n\n\n\n\nb"
        assert clean_text(text) == "a\n\nb"

    def test_strips_whitespace(self):
        assert clean_text("  hello  ") == "hello"

    def test_removes_self_closing_tags(self):
        text = "before <system-reminder foo/> after"
        assert clean_text(text) == "before  after"


class TestIsTrivial:
    def test_trivial_inputs(self):
        assert is_trivial("y") is True
        assert is_trivial("はい") is True
        assert is_trivial("OK") is True
        assert is_trivial("  続けて  ") is True

    def test_non_trivial(self):
        assert is_trivial("Pythonの質問") is False
        assert is_trivial("") is False


class TestExtractText:
    def test_string(self):
        assert extract_text("hello") == "hello"

    def test_list_with_text_blocks(self):
        content = [
            {"type": "text", "text": "part1"},
            {"type": "tool_use", "name": "foo"},
            {"type": "text", "text": "part2"},
        ]
        assert extract_text(content) == "part1\npart2"

    def test_empty_list(self):
        assert extract_text([]) == ""

    def test_other_type(self):
        assert extract_text(42) == ""


class TestExtractConversations:
    def test_simple_pair(self):
        messages = [
            {"message": {"role": "user", "content": "Q1"}},
            {"message": {"role": "assistant", "content": "A1"}},
        ]
        pairs = extract_conversations(messages)
        assert len(pairs) == 1
        q, a, ts = pairs[0]
        assert q == "Q1"
        assert a == "A1"
        assert ts is None

    def test_multiple_pairs(self):
        messages = [
            {"message": {"role": "user", "content": "Q1"}},
            {"message": {"role": "assistant", "content": "A1"}},
            {"message": {"role": "user", "content": "Q2"}},
            {"message": {"role": "assistant", "content": "A2"}},
        ]
        pairs = extract_conversations(messages)
        assert len(pairs) == 2

    def test_skips_compact_summary(self):
        messages = [
            {"message": {"role": "user", "content": "Q1"}, "isCompactSummary": True},
            {"message": {"role": "assistant", "content": "A1"}},
        ]
        pairs = extract_conversations(messages)
        assert len(pairs) == 0

    def test_tool_loop_combines_answers(self):
        messages = [
            {"message": {"role": "user", "content": "質問"}},
            {
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "調べます"},
                        {"type": "tool_use", "name": "Read"},
                    ],
                }
            },
            {
                "message": {
                    "role": "user",
                    "content": [{"type": "tool_result", "content": "data"}],
                }
            },
            {"message": {"role": "assistant", "content": "結果はこうです"}},
        ]
        pairs = extract_conversations(messages)
        assert len(pairs) == 1
        assert "調べます" in pairs[0][1]
        assert "結果はこうです" in pairs[0][1]


class TestParseJsonl:
    def test_parse(self, sample_jsonl):
        messages = parse_jsonl(sample_jsonl)
        assert len(messages) == 4

    def test_skip_invalid_json(self, tmp_path):
        path = tmp_path / "bad.jsonl"
        path.write_text('{"valid": true}\nnot json\n{"also": "valid"}\n')
        messages = parse_jsonl(path)
        assert len(messages) == 2


class TestChunkSession:
    @patch("kakolog.chunker._get_tagger")
    def test_chunk_session(self, mock_tagger, sample_jsonl):
        fake_tagger = MagicMock()
        fake_node = MagicMock()
        fake_node.surface = ""
        fake_node.next = None
        fake_tagger.parseToNode.return_value = fake_node
        mock_tagger.return_value = fake_tagger

        from kakolog.chunker import chunk_session

        chunks = chunk_session(sample_jsonl)
        assert isinstance(chunks, list)
        for chunk in chunks:
            assert isinstance(chunk, TurnChunk)
