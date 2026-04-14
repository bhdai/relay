"""Tests for relay.tools.schema — ToolSchema model."""

from unittest.mock import MagicMock

from relay.tools.schema import ToolSchema


class TestToolSchema:
    def test_from_tool_with_pydantic_schema(self):
        mock_tool = MagicMock()
        mock_tool.name = "read_file"
        mock_tool.description = "Read a file"

        # Simulate a Pydantic schema class with model_json_schema.
        mock_schema_cls = MagicMock()
        mock_schema_cls.model_json_schema.return_value = {
            "type": "object",
            "properties": {"path": {"type": "string"}},
        }
        mock_tool.tool_call_schema = mock_schema_cls

        schema = ToolSchema.from_tool(mock_tool)

        assert schema.name == "read_file"
        assert schema.description == "Read a file"
        assert schema.parameters == {
            "type": "object",
            "properties": {"path": {"type": "string"}},
        }

    def test_from_tool_with_dict_schema(self):
        mock_tool = MagicMock()
        mock_tool.name = "write_file"
        mock_tool.description = "Write a file"
        mock_tool.tool_call_schema = {
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
        }

        schema = ToolSchema.from_tool(mock_tool)

        assert schema.name == "write_file"
        assert schema.parameters == mock_tool.tool_call_schema

    def test_from_tool_with_none_schema(self):
        mock_tool = MagicMock()
        mock_tool.name = "noop"
        mock_tool.description = "No-op"
        mock_tool.tool_call_schema = None

        schema = ToolSchema.from_tool(mock_tool)

        assert schema.parameters == {"type": "object", "properties": {}}

    def test_model_dump_round_trip(self):
        schema = ToolSchema(
            name="tool",
            description="A tool",
            parameters={"type": "object"},
        )
        data = schema.model_dump()
        restored = ToolSchema.model_validate(data)
        assert restored == schema
